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

"""Auto-update subsystem.

Owns version detection, the PyPI-style update probe, the on-disk
``latest_version`` cache, and the upgrade-banner UX. The CLI's global
callback (``cli.py``) calls ``check_for_updates`` once per day and
``maybe_show_cached_banner`` on every other invocation; the
``colab update`` Typer command (``commands/utility.py``) delegates to
``check_for_updates``.
"""

import json
import logging
import os
import platform
import subprocess
import urllib.request
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as installed_version
from packaging.version import InvalidVersion, Version
from typing import Optional

import typer

from colab_cli.common import state
from colab_cli.state import Settings

logger = logging.getLogger(__name__)

# Distribution name of THIS build. It is a custom fork, not the upstream
# package, and it must never be silently replaced by the upstream one.
PYPI_PACKAGE_NAME = "colab-tool"

# The upstream project this build is based on. Its PyPI release is what the
# update probe looks at, purely as information -- installing it would throw
# away every custom change (multi-account support, the Windows port, the
# agent-facing commands).
UPSTREAM_PACKAGE_NAME = "google-colab-cli"

# A human-readable name for this build, shown in the help and the banner.
BUILD_NAME = "colab-tool"
BUILD_DESCRIPTION = "custom Colab CLI build (itzrnvr/google-colab-cli)"


# ---------- Version detection -------------------------------------------


def is_self_install_supported() -> bool:
    """Return True if self-install (--install) is supported on the current platform."""
    return platform.system() in ("Linux", "Darwin")


def get_app_version() -> str:
    """Return the installed package version, falling back to the git short hash."""
    try:
        return installed_version(PYPI_PACKAGE_NAME)
    except (PackageNotFoundError, InvalidVersion):
        pass

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
        ).strip()
    except Exception:
        return "unknown"


# ---------- Source fetchers ---------------------------------------------


def _parse_version(payload: Optional[dict]) -> Optional[str]:
    """Returns ``info.version`` from a PyPI-style payload, or None."""
    return (payload or {}).get("info", {}).get("version")


def _fetch_pypi(url: str, quiet: bool) -> Optional[dict]:
    """Fetches and parses the PyPI-style JSON document at ``url``."""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        if not quiet:
            typer.echo(f"[colab] Warning: Failed to fetch update info: {e}")
        return None


# ---------- Version comparison ------------------------------------------


def _is_newer(candidate: Optional[str], current: str) -> bool:
    """True when ``candidate`` strictly exceeds ``current`` (PEP 440)."""
    if not candidate:
        return False
    try:
        return Version(candidate) > Version(current)
    except InvalidVersion:
        return candidate != current


# ---------- UX ----------------------------------------------------------


def announce_upgrade(
    latest: str,
    current: str,
    install_cmd: str,
    *,
    show_disable_hint: bool = False,
    err: bool = False,
) -> None:
    """Print the upgrade banner.

    For this custom build the banner is informational only: the version
    being reported is the UPSTREAM project's, and installing it would throw
    away every custom change. The banner therefore says which build is
    running and how to update *this* build.

    ``show_disable_hint`` controls whether the trailing line that explains
    how to silence the auto-check is included. It is only added when the
    banner is shown unsolicited (the daily background fetch and the cached
    banner on subsequent invocations); explicit ``colab update`` calls
    omit it because the user already opted in to seeing the result.

    ``err`` sends the banner to stderr. The unsolicited background banner uses
    this so that stdout stays machine-parseable: an agent piping
    ``colab exec --json`` into a JSON parser must not have an upgrade notice
    injected into the stream.
    """
    typer.echo(
        f"\n[colab] {BUILD_NAME} is a {BUILD_DESCRIPTION}.",
        err=err,
    )
    typer.echo(
        f"[colab] Upstream {UPSTREAM_PACKAGE_NAME} has a newer release "
        f"available: {latest} (current: {current}).",
        err=err,
    )
    typer.echo(
        "[colab] To update THIS build: `colab-tool update --install` "
        "(pulls the fork and reinstalls it). Installing the upstream package "
        "would replace the custom build.",
        err=err,
    )
    if show_disable_hint:
        typer.echo(
            "[colab] To silence this check, set "
            '"enable_update_check": false in '
            "~/.config/colab-cli/settings.json",
            err=err,
        )
    typer.echo("", err=err)


# ---------- Orchestration -----------------------------------------------


def _get_install_command() -> str:
    """Return the recommended installation command based on the environment.

    For this custom build the command must reinstall *this* project, never
    the upstream package -- installing that would replace the custom build
    with the stock one.
    """
    import sys
    from itertools import pairwise
    from pathlib import Path

    checkout = find_source_checkout()
    if checkout:
        return f"pip install -e {checkout}"

    # Path.parts is separator-aware: backslashes on Windows, so a raw
    # "/uv/tools/" substring never matches "%APPDATA%\\uv\\tools\\...".
    in_uv_tool = ("uv", "tools") in pairwise(Path(sys.executable).parts)
    if is_self_install_supported() and in_uv_tool:
        return f"uv tool install -U {PYPI_PACKAGE_NAME}"
    return f"pip install --upgrade {PYPI_PACKAGE_NAME}"


def check_for_updates(quiet: bool = False) -> None:
    """Check PyPI for updates and print a message if a new version is available.

    The disable-hint is appended to the banner only when ``quiet`` is True
    (the daily background fetch); explicit ``colab update`` invocations
    (``quiet=False``) omit it because the user asked for the check.
    """
    settings = state.settings_store.load()
    current = get_app_version()

    try:
        pypi = _fetch_pypi(settings.update_url, quiet)
        pypi_v = _parse_version(pypi)

        if _is_newer(pypi_v, current):
            announce_upgrade(
                pypi_v,
                current,
                _get_install_command(),
                show_disable_hint=quiet,
                # The unsolicited background check must not pollute stdout.
                err=quiet,
            )
        elif not quiet:
            suffix = f", latest: {pypi_v}" if pypi_v else ""
            typer.echo(f"[colab] Colab CLI is up to date (version: {current}{suffix}).")

        # Cache the highest observed version; never downgrade.
        cached = settings.latest_version or "0"
        if _is_newer(pypi_v, cached):
            settings.latest_version = pypi_v

        settings.last_check = datetime.now(timezone.utc)
        state.settings_store.save(settings)

    except Exception as e:
        if not quiet:
            typer.echo(f"[colab] Failed to check for updates: {e}")


# ---------- Background hooks (called from cli.py) -----------------------


def _is_throttled(settings: Settings, *, now: Optional[datetime] = None) -> bool:
    """True if the once-per-day fetch should be skipped."""
    if settings.last_check is None:
        return False
    now = now or datetime.now(timezone.utc)
    return (now - settings.last_check).days < 1


def maybe_show_cached_banner(settings: Settings) -> None:
    """Print the cached upgrade banner if the cache reports a newer version.

    Called from the global CLI callback when the daily fetch is throttled.
    The banner uses a generic ``colab update`` install hint because the
    cache does not record which source supplied the version; the disable
    hint is shown because this is unsolicited output.
    """
    if not settings.latest_version:
        return
    current = get_app_version()
    if not _is_newer(settings.latest_version, current):
        return
    announce_upgrade(
        settings.latest_version,
        current,
        "colab update",
        show_disable_hint=True,
        err=True,
    )


def run_background_check() -> None:
    """Entry point for the global CLI callback.

    Performs either the daily fetch (which writes the cache) or, if
    throttled, surfaces the cached banner. Honors the
    ``enable_update_check`` master switch.
    """
    settings = state.settings_store.load()
    if not settings.enable_update_check:
        return
    if _is_throttled(settings):
        maybe_show_cached_banner(settings)
    else:
        check_for_updates(quiet=True)


# ---------- Self-install ------------------------------------------------


def find_source_checkout() -> Optional[str]:
    """The git checkout this build was installed from, if it can be found.

    The custom build is installed editable from its checkout, so the package
    directory sits inside the repository and the repository root can be found
    by walking up to the directory that holds ``.git``.
    """
    try:
        import importlib.resources

        pkg_dir = importlib.resources.files("colab_cli")
        # <checkout>/src/colab_cli -> <checkout>
        candidate = os.path.abspath(
            os.path.join(str(pkg_dir), os.pardir, os.pardir)
        )
        while candidate and candidate != os.path.dirname(candidate):
            if os.path.isdir(os.path.join(candidate, ".git")):
                return candidate
            candidate = os.path.dirname(candidate)
    except Exception as e:  # pragma: no cover
        logger.debug(f"Could not locate the source checkout: {e}")
    return None


def self_install() -> None:
    """Update THIS build in place: pull the fork and reinstall it.

    Never installs the upstream package -- that would silently replace the
    custom build with the stock one and lose every local change.
    """
    import sys

    checkout = find_source_checkout()
    if not checkout:
        typer.echo(
            "[colab] This is a custom build and its source checkout could not "
            "be located, so it will not be updated automatically.",
            err=True,
        )
        typer.echo(
            f"[colab] Do NOT run `pip install -U {UPSTREAM_PACKAGE_NAME}` -- "
            "that would replace the custom build with the upstream one.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"[colab] Updating the custom build from {checkout}")
    pull = subprocess.run(
        ["git", "-C", checkout, "pull", "--ff-only"],
        capture_output=True,
        encoding="utf-8",
    )
    typer.echo(pull.stdout.strip() or pull.stderr.strip())
    if pull.returncode != 0:
        typer.echo(
            "[colab] git pull failed -- resolve it in the checkout and retry.",
            err=True,
        )
        raise typer.Exit(code=pull.returncode)

    cmd = [sys.executable, "-m", "pip", "install", "-e", checkout]
    typer.echo(f"[colab] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)
