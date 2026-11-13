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

"""``colab-tool doctor`` -- one command that says whether the CLI can work, and if
not, exactly what to do about it.

An agent that is about to rent a GPU should not have to discover its own
problems one confusing error at a time. This command runs every check up
front and answers with a verdict plus a numbered list of next actions, so
even a weak model can go from "broken" to "training" in one or two steps.
"""

import json
import os
import shutil
import sys
from typing import List

import typer
from typing_extensions import Annotated

from colab_cli.accounts import AccountStore

OK = "ok"
WARN = "warn"
FAIL = "fail"


class Check:
    def __init__(self, name: str, status: str, detail: str, fix: str = ""):
        self.name = name
        self.status = status
        self.detail = detail
        self.fix = fix

    def as_dict(self) -> dict:
        return {
            "check": self.name,
            "status": self.status,
            "detail": self.detail,
            "fix": self.fix,
        }


def _check_interpreter() -> List[Check]:
    """Is the running interpreter the one the agent expects?"""
    out = []
    exe = os.path.basename(sys.executable)
    out.append(
        Check(
            "interpreter",
            OK,
            f"{sys.executable} (python {sys.version.split()[0]})",
        )
    )
    # The fork guards its POSIX imports, so `import colab_cli` must work
    # without any stdlib shim.
    try:
        import colab_cli  # noqa: F401

        out.append(
            Check(
                "colab_cli import",
                OK,
                "the installed colab_cli imports cleanly",
            )
        )
    except Exception as e:
        out.append(
            Check(
                "colab_cli import",
                FAIL,
                f"importing colab_cli failed: {e}",
                "Reinstall the CLI: pip install -e <repo>  "
                "(on Windows a stale upstream build crashes with "
                "'No module named termios')",
            )
        )

    # A hand-written termios.py in the stdlib is a red flag: it means an
    # earlier workaround masked a broken install.
    for libdir in sys.path[:3]:
        if not libdir or not os.path.isdir(libdir):
            continue
        shim = os.path.join(libdir, "termios.py")
        if os.path.exists(shim):
            out.append(
                Check(
                    "stdlib termios shim",
                    WARN,
                    f"{shim} is a hand-written shim, not the real module",
                    "Delete it and install the CLI properly -- the shim "
                    "makes imports work but leaves termios.tcgetattr() broken",
                )
            )
            break
    return out


def _check_path() -> List[Check]:
    """Which `colab` will actually run?"""
    out = []
    found = shutil.which("colab")
    if not found:
        out.append(
            Check(
                "colab on PATH",
                FAIL,
                "no 'colab' executable on PATH",
                "Add the directory holding colab.exe to PATH, or call it by "
                "its full path",
            )
        )
        return out
    out.append(Check("colab on PATH", OK, found))
    return out


def _check_accounts() -> List[Check]:
    out = []
    store = AccountStore()
    active = store.active()
    usable = store.usable_accounts()
    if not usable:
        out.append(
            Check(
                "accounts",
                FAIL,
                "no Google account is signed in",
                "Run: colab-tool account add default",
            )
        )
        return out
    emails = []
    for label in usable:
        email = store.meta(label).get("email") or "(email unknown)"
        emails.append(f"{label} ({email})")
    detail = ", ".join(emails)
    out.append(
        Check(
            "accounts",
            OK,
            f"{len(usable)} signed in; active: {active or '(none)'} -- {detail}",
        )
    )
    if len(usable) > 1:
        out.append(
            Check(
                "account failover",
                OK,
                "a second account is available, so a quota refusal on the "
                "active account can rotate automatically",
            )
        )
    else:
        out.append(
            Check(
                "account failover",
                WARN,
                "only one account is signed in, so a quota refusal has "
                "nowhere to rotate to",
                "Run: colab-tool account add backup   (a second account means "
                "'colab-tool new' can recover from a spent quota)",
            )
        )
    return out


def _check_auth() -> List[Check]:
    """Can we actually talk to the backend?"""
    from colab_cli.common import state

    try:
        assignments = state.client.list_assignments()
    except Exception as e:
        return [
            Check(
                "backend auth",
                FAIL,
                f"could not list assignments: {e}",
                "Run: colab-tool whoami   (it prints the active identity and "
                "scopes). If it 401s, re-mint the token with "
                "'colab-tool account add default'.",
            )
        ]
    return [
        Check(
            "backend auth",
            OK,
            f"authenticated; {len(assignments)} assignment(s) on this account",
        )
    ]


def _check_sessions() -> List[Check]:
    """Are there live sessions, and are they actually reachable?"""
    from colab_cli.common import state

    try:
        sessions, assignments = state.sync_sessions()
    except Exception as e:
        return [Check("sessions", WARN, f"could not sync sessions: {e}")]

    if not sessions:
        return [
            Check(
                "sessions",
                OK,
                "no local sessions (a fresh start)",
            )
        ]

    out = []
    for name, s in sessions.items():
        detail = f"{name}: {s.accelerator} at {s.endpoint}"
        if s.account:
            detail += f" (account {s.account})"
        out.append(Check(f"session '{name}'", OK, detail))
    return out


def run_checks(include_live: bool = True) -> List[Check]:
    checks: List[Check] = []
    checks += _check_interpreter()
    checks += _check_path()
    checks += _check_accounts()
    if include_live:
        checks += _check_auth()
        checks += _check_sessions()
    return checks


def doctor(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the report as JSON"),
    ] = False,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help="Skip the checks that talk to Google (no network, no quota)",
        ),
    ] = False,
):
    """Check that this machine can drive Colab, and say what to do next."""
    checks = run_checks(include_live=not offline)

    failures = [c for c in checks if c.status == FAIL]
    warnings = [c for c in checks if c.status == WARN]

    if json_output:
        payload = {
            "ok": not failures,
            "checks": [c.as_dict() for c in checks],
            "next": [c.fix for c in checks if c.status == FAIL and c.fix],
        }
        typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(code=0 if not failures else 1)

    typer.echo("")
    for c in checks:
        mark = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]"}[c.status]
        typer.echo(f" {mark} {c.name}: {c.detail}")

    typer.echo("")
    if failures:
        typer.echo(
            f"NOT READY: {len(failures)} problem(s) to fix, "
            f"{len(warnings)} warning(s).",
            err=True,
        )
        typer.echo("Do this next, in order:", err=True)
        for i, c in enumerate([c for c in checks if c.status == FAIL and c.fix], 1):
            typer.echo(f"  {i}. {c.fix}", err=True)
        typer.echo("")
        raise typer.Exit(code=1)

    if warnings:
        typer.echo(
            f"READY, with {len(warnings)} warning(s).", err=True
        )
        for i, c in enumerate([c for c in checks if c.status == WARN and c.fix], 1):
            typer.echo(f"  {i}. {c.fix}", err=True)
        typer.echo("")
        return

    typer.echo("READY. You can rent a runtime: colab-tool new -s <name> --gpu T4")
    typer.echo("")


def register(root: typer.Typer):
    root.command(name="doctor")(doctor)
