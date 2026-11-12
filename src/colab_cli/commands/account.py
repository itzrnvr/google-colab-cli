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

"""``colab account`` — manage multiple signed-in Google accounts.

Colab applies per-account limits (daily compute quota, concurrent GPU
assignments). When one account is exhausted, ``colab new --gpu T4`` fails with
HTTP 412 even though another signed-in account still has headroom. These
commands let several accounts be signed in side by side so the CLI (or an
agent driving it) can rotate to a healthy one.

    colab account list                # show stored accounts + active one
    colab account add work            # sign in an additional account
    colab account use work            # make it the active account
    colab account rm work             # forget it
    colab account current             # print the active label
"""

from typing import Annotated, Optional

import typer

from colab_cli.accounts import AccountError, AccountStore

app = typer.Typer(
    help=(
        "Manage multiple signed-in Google accounts.\n\n"
        "Colab enforces per-account limits. Keep several accounts signed in "
        "and the CLI will rotate to a healthy one when the active account is "
        "refused an allocation (HTTP 412)."
    ),
    no_args_is_help=True,
)


def _store() -> AccountStore:
    return AccountStore()


@app.command(name="list")
def list_accounts(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the account list as JSON"),
    ] = False,
    resolve: Annotated[
        bool,
        typer.Option(
            "--resolve",
            help=(
                "Look up the Google account email for accounts whose email is "
                "not known yet (needs a network call)."
            ),
        ),
    ] = True,
):
    """List the accounts signed in on this machine."""
    store = _store()
    active = store.active()
    rows = []
    for label in store.labels():
        meta = store.meta(label)
        email = meta.get("email", "")
        if resolve and not email and store.has(label):
            from colab_cli.auth import resolve_stored_account_email

            email = resolve_stored_account_email(label, store=store) or ""
        rows.append(
            {
                "label": label,
                "email": email,
                "active": label == active,
                "added_at": meta.get("added_at", ""),
                "last_used": meta.get("last_used", ""),
                "has_token": store.has(label),
            }
        )

    if json_output:
        import json as _json

        typer.echo(_json.dumps(rows, indent=2))
        return

    if not rows:
        typer.echo(
            "No accounts signed in. Run `colab account add <label>` to sign in.",
            err=True,
        )
        return

    width = max(len(r["label"]) for r in rows)
    for r in rows:
        marker = "*" if r["active"] else " "
        email = r["email"] or "(unknown)"
        extra = "" if r["has_token"] else "  [no token on disk]"
        typer.echo(f" {marker} {r['label']:<{width}}  {email}{extra}")
    typer.echo("")
    typer.echo(
        f"Active: {active or '(none)'}   "
        "(use `colab account use <label>` to switch, "
        "or `colab --account <label>` for a single command)"
    )


@app.command(name="add")
def add_account(
    label: Annotated[
        str,
        typer.Argument(help="Label for the account, e.g. 'work' or 'personal'"),
    ],
    use: Annotated[
        bool,
        typer.Option("--use", help="Make this the active account after signing in"),
    ] = True,
    from_file: Annotated[
        Optional[str],
        typer.Option(
            "--from-file",
            help=(
                "Import an existing OAuth token file instead of running the "
                "sign-in flow (e.g. a token.json that was backed up)."
            ),
        ),
    ] = None,
):
    """Sign in an additional Google account and store it under LABEL."""
    store = _store()
    try:
        if from_file:
            from colab_cli.auth import import_account_token

            email = import_account_token(label, from_file, store=store)
        else:
            from colab_cli.auth import sign_in_account

            email = sign_in_account(label, config_path=None, store=store)
    except AccountError as e:
        typer.echo(f"[colab] {e}", err=True)
        raise typer.Exit(2)
    except FileNotFoundError as e:
        typer.echo(f"[colab] {e}", err=True)
        raise typer.Exit(2)

    if use:
        store.set_active(label)
    typer.echo(f"[colab] Signed in '{label}' ({email or 'unknown account'}).")
    if not use:
        typer.echo(f"[colab] Activate it later with `colab account use {label}`.")


@app.command(name="use")
def use_account(
    label: Annotated[str, typer.Argument(help="Account label to activate")],
):
    """Make LABEL the active account."""
    store = _store()
    try:
        store.set_active(label)
    except AccountError as e:
        typer.echo(f"[colab] {e}", err=True)
        raise typer.Exit(2)
    meta = store.meta(label)
    email = meta.get("email") or "(unknown)"
    typer.echo(f"[colab] Active account is now '{label}' ({email}).")


@app.command(name="rm")
def remove_account(
    label: Annotated[str, typer.Argument(help="Account label to remove")],
):
    """Forget the account stored under LABEL (deletes its token)."""
    store = _store()
    if not store.known(label):
        typer.echo(f"[colab] No account named '{label}'.", err=True)
        raise typer.Exit(2)
    store.remove(label)
    typer.echo(f"[colab] Removed account '{label}'.")


@app.command(name="current")
def current_account(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the active account as JSON"),
    ] = False,
):
    """Print the label of the active account."""
    store = _store()
    active = store.active()
    if json_output:
        import json as _json

        meta = store.meta(active) if active else {}
        typer.echo(
            _json.dumps(
                {"label": active, "email": meta.get("email", "")}, indent=2
            )
        )
        return
    if not active:
        typer.echo("(none)", err=True)
        raise typer.Exit(1)
    typer.echo(active)


def register(root: typer.Typer):
    root.add_typer(app, name="account")
