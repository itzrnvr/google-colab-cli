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

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import typer

from colab_cli.accounts import DEFAULT_ACCOUNT, AccountError, AccountStore
from colab_cli.auth import AuthProvider, get_credentials
from colab_cli.client import Client, Prod, RuntimeProxyInfo
from colab_cli.history import HistoryLogger
from colab_cli.state import SessionState, StateStore, SettingsStore

logger = logging.getLogger(__name__)

# Headroom so a token doesn't expire mid-command.
TOKEN_REFRESH_MARGIN = timedelta(minutes=5)


def _apply_proxy_info(s: SessionState, info: RuntimeProxyInfo):
    s.token = info.token
    s.url = info.url
    s.token_expires_at = info.expires_at()


class State:
    def __init__(self):
        self.client_oauth_config = os.path.expanduser("~/.colab-cli-oauth-config.json")
        self.config_path = None
        self.logtostderr = False
        self.auth_provider = AuthProvider.OAUTH2
        # Explicit account override (``--account``). ``None`` means "use the
        # active account from the account index".
        self.account = None
        self._client = None
        self._store = None
        self._settings_store = None
        self._history = None
        self._sessions = None
        self._accounts = None

    @property
    def accounts(self) -> AccountStore:
        if self._accounts is None:
            self._accounts = AccountStore()
        return self._accounts

    def current_account(self) -> str:
        """The account label that will be used for the next request."""
        return self.account or self.accounts.active() or DEFAULT_ACCOUNT

    def use_account(self, label: str, make_active: bool = True) -> None:
        """Sign in as ``label`` for the remainder of this invocation.

        Rebuilds the API client so the next request uses the new credentials.
        """
        self.account = label
        self._client = None
        if make_active:
            try:
                self.accounts.set_active(label)
            except AccountError:
                # Not every label has a token yet; the explicit override still
                # applies for this invocation.
                pass

    def rotate_account(self, exclude: Optional[str] = None) -> Optional[str]:
        """Switch to the next stored account that still has a token.

        Returns the new label, or ``None`` when there is no other account.
        """
        current = exclude or self.current_account()
        nxt = self.accounts.next_account(exclude=current)
        if not nxt:
            return None
        self.use_account(nxt)
        self.accounts.advance_rotation(nxt)
        return nxt

    @property
    def store(self):
        if self._store is None:
            self._store = StateStore(self.config_path)
        return self._store

    @property
    def settings_store(self):
        if self._settings_store is None:
            # We don't currently allow overriding settings path via CLI,
            # but we could if needed. For now, use default.
            self._settings_store = SettingsStore()
        return self._settings_store

    @property
    def history(self):
        if self._history is None:
            self._history = HistoryLogger()
        return self._history

    @property
    def client(self):
        if self._client is None:
            creds = get_credentials(
                self.client_oauth_config,
                provider=self.auth_provider,
                account=self.account,
            )
            self._client = Client(Prod(), creds)
        return self._client

    def client_for(self, account: str) -> Client:
        """Build an API client for ``account`` without changing any state.

        Used to look up runtimes that belong to a different account than the
        one currently active.
        """
        creds = get_credentials(
            self.client_oauth_config,
            provider=self.auth_provider,
            account=account,
        )
        return Client(Prod(), creds)

    def prune_session(self, name: str):
        """Removes a session from local state."""
        self.store.remove(name)
        if self._sessions and name in self._sessions:
            del self._sessions[name]
        self.history.log_event(name, "session_terminated", {"reason": "pruned"})

    def get_session(
        self, name: str, ignore_missing_session: bool = False
    ) -> Optional[SessionState]:
        """Load a session, refreshing its runtime proxy token if it's near expiry.

        A session is missing if it's unknown locally or its assignment is gone
        server-side (in which case it's pruned). Missing sessions print an error
        and exit, unless ignore_missing_session is set, in which case this
        returns None.
        """
        s = self.store.get(name)
        if s and (
            not s.token_expires_at
            or s.token_expires_at - datetime.now(timezone.utc) <= TOKEN_REFRESH_MARGIN
        ):
            by_endpoint = {
                a.endpoint: a for a in self._assignments_for(s.account)
            }
            if s.endpoint in by_endpoint:
                _apply_proxy_info(s, by_endpoint[s.endpoint].runtime_proxy_info)
                self.store.add(s)
            else:
                self.prune_session(name)
                s = None

        if s is None and not ignore_missing_session:
            typer.echo(
                f"[colab] Session '{name}' not found. It is unknown locally "
                "or its runtime is gone. List what is running with "
                "`colab sessions`, or create one with `colab new -s <name>`.",
                err=True,
            )
            raise typer.Exit(1)
        return s

    def _assignments_for(self, account: Optional[str] = None):
        """List server-side assignments for ``account``.

        ``account`` of ``None`` means "the currently active account". Sessions
        created under a different account are invisible to the active
        account's listing, so they must be queried separately.
        """
        if account and account != self.current_account():
            try:
                return self.client_for(account).list_assignments()
            except Exception as e:
                logger.debug(f"Could not list assignments for account {account}: {e}")
                return []
        return self.client.list_assignments()

    def sync_sessions(self):
        if self._sessions is not None:
            return self._sessions, self.client.list_assignments()

        # Check local store first. If it's empty, we don't necessarily need to hit the backend
        # unless we are specifically looking for server-side assignments (e.g. 'colab sessions').
        local_sessions = self.store.list()
        if not local_sessions:
            self._sessions = {}
            # We still need to return assignments for 'colab sessions' to work
            # But we only trigger client creation (and thus auth) if we have to.
            try:
                assignments = self.client.list_assignments()
            except SystemExit:
                # If auth fails, we just return empty assignments
                assignments = []
            return self._sessions, assignments

        assignments = self.client.list_assignments()
        by_endpoint = {a.endpoint: a for a in assignments}

        # Sessions owned by a different account (created after a failover) are
        # not in the active account's listing; look them up under their owner.
        current = self.current_account()
        other_accounts = {
            s.account
            for s in local_sessions.values()
            if s.account and s.account != current
        }
        for acct in other_accounts:
            for a in self._assignments_for(acct):
                by_endpoint.setdefault(a.endpoint, a)

        self._sessions = local_sessions
        pruned = 0
        for name, s in list(self._sessions.items()):
            if s.endpoint not in by_endpoint:
                self.prune_session(name)
                pruned += 1
            else:
                _apply_proxy_info(s, by_endpoint[s.endpoint].runtime_proxy_info)
                self.store.add(s)

        if pruned > 0:
            # A notification the agent needs: a runtime went away (Colab
            # reclaims idle VMs, or the VM was stopped). It goes to stderr
            # so it never corrupts a --json stream on stdout, and it says
            # what to do about it.
            typer.echo(
                f"[colab] Notice: {pruned} session(s) were pruned because "
                "their runtime is gone (Colab reclaims idle VMs). Any work "
                "that was running there is lost. Create a new one with "
                "`colab new -s <name> --gpu T4`.",
                err=True,
            )

        return self._sessions, assignments

    def resolve_session(self, session_name: Optional[str]) -> str:
        if session_name:
            return session_name

        # Check local store first to avoid hitting the backend (and triggering auth) if we don't have to
        local_sessions = self.store.list()
        if not local_sessions:
            typer.echo(
                "[colab] Error: No active sessions found. Create one with 'colab new'.",
                err=True,
            )
            raise typer.Exit(1)

        # If we have local sessions, we need to sync to make sure they are still valid.
        # This will trigger auth if valid credentials are not present.
        sessions, _ = self.sync_sessions()
        active_names = list(sessions.keys())

        if len(active_names) == 1:
            name = active_names[0]
            typer.echo(f"[colab] Using unique session '{name}'.", err=True)
            return name
        elif len(active_names) > 1:
            typer.echo(
                f"[colab] Error: Multiple active sessions found. Specify one with -s: {', '.join(active_names)}",
                err=True,
            )
            raise typer.Exit(1)
        else:
            typer.echo(
                "[colab] Error: No active sessions found. Create one with 'colab new'.",
                err=True,
            )
            raise typer.Exit(1)


state = State()


def setup_logging(log_to_stderr: bool):
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    requests_log = logging.getLogger("urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

    log_dir = os.path.expanduser("~/.config/colab-cli")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "colab.log"))
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    if log_to_stderr:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(stream_handler)
