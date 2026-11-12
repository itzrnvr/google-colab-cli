# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import json
import logging
from importlib import resources
import os
import warnings
from typing import Optional

import google.auth
import typer
from google.auth.transport import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from colab_cli.accounts import (
    DEFAULT_ACCOUNT,
    LEGACY_TOKEN_PATH,
    AccountError,
    AccountStore,
    validate_label,
)

logger = logging.getLogger(__name__)


class AuthProvider(str, enum.Enum):
    """Authentication strategy for talking to the Colab backend.

    Values are the lowercase strings accepted by the global ``--auth`` flag.
    """

    OAUTH2 = "oauth2"
    ADC = "adc"


# Standard Scopes for Colab and Drive (Public Auth)
PUBLIC_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]


# Legacy single-account token path. Kept as an alias of the canonical
# definition in `colab_cli.accounts` so both spellings are byte-identical.
TOKEN_CONFIG_PATH = LEGACY_TOKEN_PATH

# Remote copy-paste OAuth flow.
#
# We deliberately do NOT use a localhost redirect (`run_local_server`) or the
# out-of-band (OOB) redirect `urn:ietf:wg:oauth:2.0:oob`. OOB was blocked by
# Google in 2022 ("The out-of-band (OOB) flow has been blocked in order to
# keep users secure") and a localhost server is environment-dependent (fails
# on headless/remote/container hosts, requires an auto-openable browser, etc.).
#
# Instead we use the same mechanism `gcloud auth application-default login`
# uses: a real registered HTTPS landing page that displays the authorization
# code for the user to copy & paste, combined with the `token_usage=remote`
# consent parameter. This works identically in local and remote environments.
#
# The landing page below is registered to Google's cloud-SDK OAuth client
# (`764086051850-...`), which is also the client shipped in
# `colab_cli/oauth_config.json`; reusing another client id with this redirect
# yields `redirect_uri_mismatch`.
REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"


def _run_remote_flow(client_config: dict) -> Credentials:
    """Run the remote copy-paste OAuth2 flow.

    Prints an authorization URL, waits for the user to sign in and paste back
    the authorization code shown on Google's landing page, then exchanges the
    code for credentials. See ``REMOTE_REDIRECT_URI`` for why this is preferred
    over a localhost server or the blocked OOB flow.
    """
    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    auth_url, _ = flow.authorization_url(prompt="consent", token_usage="remote")

    typer.echo("\nTo authorize colab-cli, visit this URL in any browser:\n", err=True)
    typer.echo("  " + auth_url + "\n", err=True)
    typer.echo("After approving, Google will display an authorization code.", err=True)
    code = input("Enter the authorization code: ").strip()

    flow.fetch_token(code=code)
    return flow.credentials


def _decode_jwt_claims(token: str) -> dict:
    """Best-effort decode of a JWT payload (no signature verification)."""
    try:
        import base64

        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
    except Exception:
        return {}


def resolve_account_email(creds: Credentials) -> Optional[str]:
    """Best-effort lookup of the signed-in Google account's email address.

    Used to label accounts so ``colab account list`` can show which Google
    account a token belongs to. Prefers the ``email`` claim in the OIDC
    ``id_token``; falls back to the OAuth2 userinfo endpoint.
    """
    try:
        id_token = getattr(creds, "id_token", None)
        if id_token:
            email = _decode_jwt_claims(id_token).get("email")
            if isinstance(email, str) and email:
                return email
    except Exception as e:  # pragma: no cover
        logger.debug(f"Could not read email from id_token: {e}")

    try:
        resp = requests.AuthorizedSession(creds).get(
            "https://www.googleapis.com/oauth2/v3/userinfo", timeout=15
        )
        if resp.ok:
            email = resp.json().get("email")
            if isinstance(email, str) and email:
                return email
    except Exception as e:  # pragma: no cover - network best effort
        logger.debug(f"Could not resolve account email via userinfo: {e}")
    return None


def _get_google_auth_credentials(
    config_path: str, account: Optional[str] = None
) -> Credentials:
    """
    Retrieves credentials using standard public OAuth2 flow.

    ``account`` selects which stored account to use. When ``None`` the active
    account from the account index is used, falling back to the legacy
    single-account token file.
    """
    client_config = None
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            client_config = json.load(f)
    else:
        # Last resort: try inlined config
        try:
            config_resource = resources.files("colab_cli").joinpath("oauth_config.json")
            if config_resource.is_file():
                client_config = json.loads(config_resource.read_text())
        except Exception as e:
            logger.debug(f"Failed to load inlined config: {e}")

    if not client_config:
        raise FileNotFoundError(
            f"Client OAuth config not found at {config_path} and no inlined config available. "
            "Please provide a valid path via -c/--client-oauth-config."
        )

    creds = None

    store = AccountStore()
    label = account or store.active() or DEFAULT_ACCOUNT
    token_path = store.token_path(label)

    # Ensure config directory exists for the token file
    os.makedirs(os.path.dirname(token_path), exist_ok=True)

    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(
                token_path, PUBLIC_SCOPES
            )
        except Exception as e:
            logger.warning(f"Failed to load token from {token_path}: {e}")

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.warning(f"Failed to refresh token: {e}")
                creds = None

        if not creds:
            creds = _run_remote_flow(client_config)

        # Save the credentials for the next run
        try:
            email = resolve_account_email(creds)
            store.save_token(label, json.loads(creds.to_json()), email=email)
            store.touch(label)
        except Exception as e:
            logger.error(f"Failed to save token to {token_path}: {e}")

    return creds


def sign_in_account(
    label: str,
    config_path: Optional[str] = None,
    store: Optional[AccountStore] = None,
) -> str:
    """Run the OAuth flow for a new account and store it under ``label``.

    Returns the detected Google account email (empty string when it could not
    be resolved). The token is written to the account's own file so several
    accounts can be signed in side by side.
    """
    if not config_path:
        config_path = os.path.expanduser("~/.colab-cli-oauth-config.json")
    store = store or AccountStore()

    client_config = None
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            client_config = json.load(f)
    else:
        try:
            config_resource = resources.files("colab_cli").joinpath("oauth_config.json")
            if config_resource.is_file():
                client_config = json.loads(config_resource.read_text())
        except Exception as e:
            logger.debug(f"Failed to load inlined config: {e}")

    if not client_config:
        raise FileNotFoundError(
            f"Client OAuth config not found at {config_path} and no inlined config "
            "available. Please provide a valid path via -c/--client-oauth-config."
        )

    creds = _run_remote_flow(client_config)
    email = resolve_account_email(creds) or ""
    store.save_token(label, json.loads(creds.to_json()), email=email or None)
    store.touch(label)
    return email


def resolve_stored_account_email(
    label: str,
    store: Optional[AccountStore] = None,
) -> Optional[str]:
    """Look up (and cache) the Google account email behind a stored token.

    Used by ``colab account list`` so accounts can be told apart. Resolving
    the email needs a network call, so it is done on demand rather than on
    every credential load.
    """
    store = store or AccountStore()
    known = store.meta(label).get("email")
    if known:
        return known
    token_path = store.token_path(label)
    if not os.path.exists(token_path):
        return None
    try:
        with open(token_path, "r", encoding="utf-8") as f:
            token = json.load(f)
        creds = Credentials.from_authorized_user_info(token, PUBLIC_SCOPES)
        email = resolve_account_email(creds)
    except Exception as e:
        logger.debug(f"Could not resolve the email for account {label}: {e}")
        return None
    if email:
        store.register(label, email=email)
    return email


def import_account_token(
    label: str,
    token_path: str,
    store: Optional[AccountStore] = None,
) -> str:
    """Register an existing OAuth token file under ``label``.

    Lets an already-signed-in account be added without running the OAuth
    flow again (e.g. a token file that was backed up earlier). Returns the
    detected account email, or an empty string.
    """
    store = store or AccountStore()
    with open(token_path, "r", encoding="utf-8") as f:
        token = json.load(f)

    email = ""
    try:
        creds = Credentials.from_authorized_user_info(token, PUBLIC_SCOPES)
        email = resolve_account_email(creds) or ""
    except Exception as e:
        logger.debug(f"Could not resolve the email for the imported token: {e}")

    store.save_token(label, token, email=email or None)
    store.touch(label)
    return email


def _get_adc_credentials() -> Credentials:
    """Retrieves credentials using Google Application Default Credentials.

    Honors the standard ADC discovery chain (``GOOGLE_APPLICATION_CREDENTIALS``,
    ``gcloud auth application-default login``, GCE/GKE metadata server, etc.).

    The RuntimeService at colab.pa.googleapis.com requires the `colaboratory`
    scope. Most ADC credential types (service accounts, GCE/GKE, impersonated)
    support `with_scopes`; user credentials minted by
    `gcloud auth application-default login` do not. For the latter, the user
    must run `gcloud auth application-default login` with
    `--scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory`
    (`openid` and `cloud-platform` are required by `gcloud` itself; `userinfo.email`
    is required by the session backend; `colaboratory` is retained for Colab APIs).
    """
    # `google.auth._default` emits a UserWarning when ADC user credentials
    # don't have a quota project pinned ("Your application has authenticated
    # using end user credentials from Google Cloud SDK without a quota
    # project. You might receive a 'quota exceeded' or 'API not enabled'
    # error.").
    #
    # That warning shows up on every single `colab` invocation under ADC,
    # which is pure noise. Filter it, but keep the scope as tight as possible:
    # only this exact message, only during this one call.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Your application has authenticated using end user credentials.*",
            category=UserWarning,
        )
        creds, _ = google.auth.default(scopes=list(PUBLIC_SCOPES))

    if not creds.valid:
        from google.auth import compute_engine

        if isinstance(creds, compute_engine.Credentials):
            creds = None
        else:
            logger.warning("Failed to obtain valid ADC credentials.")
            try:
                logger.warning("Trying to refresh ADC credentials")
                creds.refresh(Request())
            except Exception as e:
                logger.warning(f"Failed to refresh token: {e}")
                creds = None

    if not creds:
        typer.echo(
            "No valid default credentials found. To authenticate, run:\n\n"
            "  gcloud auth application-default login \\\n"
            "      --scopes=openid,"
            "https://www.googleapis.com/auth/cloud-platform,"
            "https://www.googleapis.com/auth/userinfo.email,"
            "https://www.googleapis.com/auth/colaboratory\n",
            err=True,
        )
        exit(1)

    # Some credential subclasses ignore the `scopes=` kwarg in `default()`
    # (e.g. user creds), so re-apply via `with_scopes` when supported.
    if getattr(creds, "requires_scopes", False):
        try:
            creds = creds.with_scopes(list(PUBLIC_SCOPES))
        except Exception as e:  # NotImplementedError for non-scopable creds.
            logger.debug(f"Could not augment ADC scopes via with_scopes: {e}")
    return creds


def get_credentials(
    config_path: Optional[str] = None,
    provider: AuthProvider = AuthProvider.OAUTH2,
    account: Optional[str] = None,
) -> requests.AuthorizedSession:
    """Unified entry point for retrieving an authorized session.

    Args:
      config_path: Path to the OAuth2 client config JSON. Only consulted when
        ``provider`` is ``OAUTH2``.
      provider: Which authentication strategy to use.
      account: Which stored account to sign in as. ``None`` means "use the
        active account" (falling back to the legacy single-token file).
    """
    if provider == AuthProvider.OAUTH2:
        if not config_path:
            config_path = os.path.expanduser("~/.colab-cli-oauth-config.json")
        if account:
            try:
                validate_label(account)
            except AccountError as e:
                raise ValueError(str(e)) from e
        creds = _get_google_auth_credentials(config_path, account=account)
    elif provider == AuthProvider.ADC:
        creds = _get_adc_credentials()
    else:
        raise ValueError(f"Unknown auth provider: {provider!r}")

    return requests.AuthorizedSession(creds)
