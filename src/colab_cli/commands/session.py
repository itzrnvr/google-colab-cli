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
import uuid
from typing import Optional
import requests
import typer
from typing_extensions import Annotated

from colab_cli.client import (
    Accelerator,
    ColabRequestError,
    HIGH_MEM_ONLY_ACCELERATORS,
    PostAssignmentResponse,
    Shape,
    TooManyAssignmentsError,
    Variant,
    resolve_assign_shape,
    shape_display_label,
)
from colab_cli.utils import get_status_code
from colab_cli.state import SessionState
from colab_cli.runtime import ColabRuntime

logger = logging.getLogger(__name__)

# How many accounts to try when the backend refuses an allocation because the
# account's quota or concurrency limit is exhausted.
MAX_ACCOUNT_ATTEMPTS = 5


def _hardware_label(accelerator: str) -> str:
    """`NONE` -> `CPU`; everything else passes through."""
    return "CPU" if accelerator == "NONE" else accelerator


def _format_session_line(
    name: str,
    endpoint: str,
    accelerator: str,
    variant: str,
    status: Optional[str] = None,
    machine_shape: Optional[str] = None,
) -> str:
    """Single source of truth for session display lines.

    Format: ``[name] endpoint | Hardware: X | Shape: Y | Variant: Z[ | Status: W]``.
    Use ``"?"`` as the name for orphaned server-side assignments with no local
    state.
    """
    parts = [
        f"[{name}] {endpoint}",
        f"Hardware: {_hardware_label(accelerator)}",
        f"Shape: {shape_display_label(machine_shape)}",
        f"Variant: {variant}",
    ]
    if status is not None:
        parts.append(f"Status: {status}")
    return " | ".join(parts)


def resolve_runtime_options(
    gpu: Optional[str] = None,
    tpu: Optional[str] = None,
    *,
    high_mem: bool = False,
) -> tuple[Variant, Accelerator, Optional[Shape]]:
    """Map CLI flags to backend variant, accelerator, and optional shape."""
    if tpu:
        variant = Variant.TPU
        accelerator = Accelerator.V5E1 if tpu.lower() == "v5e1" else Accelerator.V6E1
    elif gpu:
        variant = Variant.GPU
        mapping = {
            "a100": Accelerator.A100,
            "h100": Accelerator.H100,
            "l4": Accelerator.L4,
            "t4": Accelerator.T4,
            "g4": Accelerator.G4,
        }
        accelerator = mapping.get(gpu.lower(), Accelerator.A100)
    else:
        variant = Variant.DEFAULT
        accelerator = Accelerator.NONE

    shape = resolve_assign_shape(accelerator, high_mem=high_mem)
    return variant, accelerator, shape


def _account_label(state) -> Optional[str]:
    """The active account label, or ``None`` when it cannot be determined.

    Defensive on purpose: the account label ends up in a pydantic model field
    (``SessionState.account``) and in comparisons against other accounts, so
    a non-string value (e.g. a test double) must never reach it.
    """
    try:
        label = state.current_account()
    except Exception:
        return None
    return label if isinstance(label, str) and label else None


def _adopt_existing_session(
    name: str, variant: Variant, accelerator: Accelerator
) -> bool:
    """Reuse a matching live runtime instead of requesting a second one.

    Requesting a second GPU runtime while one is already running hits the
    account's concurrency limit (HTTP 412 ``Precondition Failed``) and burns
    wall-clock time re-provisioning. Adopting the live runtime avoids both.

    Returns ``True`` when a session is now available under some local name
    and the caller should stop.
    """
    from colab_cli.common import state

    try:
        assignments = list(state.client.list_assignments())
        matches = [a for a in assignments if a.accelerator == accelerator]
    except Exception as e:
        # The pre-flight is best effort: if the listing fails (e.g. auth not
        # settled yet, or the backend is unreachable) fall through to the
        # normal allocation path, which surfaces the real error.
        logger.debug(f"Pre-flight assignment lookup failed: {e}")
        return False

    if not matches:
        return False

    assignment = matches[0]
    local = state.store.list()

    already = next(
        (s for s in local.values() if s.endpoint == assignment.endpoint), None
    )
    if already is not None:
        typer.echo(
            f"[colab] A {accelerator.value} runtime is already running as "
            f"session '{already.name}' ({assignment.endpoint}). Reusing it. "
            "Pass --force-new to request a separate runtime.",
            err=True,
        )
        return True

    s = SessionState(
        name=name,
        token=assignment.runtime_proxy_info.token,
        url=assignment.runtime_proxy_info.url,
        endpoint=assignment.endpoint,
        token_expires_at=assignment.runtime_proxy_info.expires_at(),
        variant=assignment.variant.name,
        accelerator=assignment.accelerator.value,
        machine_shape=(
            Shape.HIGH_RAM.name
            if assignment.machine_shape == Shape.HIGH_RAM
            else Shape.STANDARD.name
        ),
        account=_account_label(state),
    )
    state.store.add(s)
    state.history.log_event(
        name,
        "session_created",
        {
            "endpoint": assignment.endpoint,
            "variant": s.variant,
            "accelerator": s.accelerator,
            "machine_shape": s.machine_shape,
            "adopted": True,
        },
    )
    typer.echo(
        f"[colab] Adopted the running {accelerator.value} runtime "
        f"{assignment.endpoint} as session '{name}'."
    )
    typer.echo("[colab] Session READY.")
    return True


def _assign_with_failover(
    variant: Variant, accelerator: Accelerator, shape: Optional[Shape]
):
    """Request an assignment, rotating accounts when the backend refuses.

    Colab enforces per-Google-account limits. When the active account's daily
    quota or GPU concurrency limit is exhausted the backend refuses the
    allocation (HTTP 412) or rejects the accelerator (HTTP 400). If another
    account is signed in, switch to it and retry so the work keeps moving.

    An exhausted account can also manifest as the allocation request *hanging*
    until the HTTP read timeout expires (observed in practice: the burned
    account times out after 120 s while a healthy account answers in
    seconds). That is treated as a refusal too, so the rotation still
    happens.
    """
    from colab_cli.common import state

    last_error = None
    for _ in range(MAX_ACCOUNT_ATTEMPTS):
        account = state.current_account()
        try:
            return state.client.assign(
                uuid.uuid4(), variant=variant, accelerator=accelerator, shape=shape
            )
        except TooManyAssignmentsError as e:
            last_error = e
            reason = "allocation refused (precondition failed)"
        except ColabRequestError as e:
            status = get_status_code(e)
            if status not in (400, 429) or accelerator == Accelerator.NONE:
                raise
            last_error = e
            reason = f"request refused (HTTP {status})"
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            last_error = e
            reason = "allocation request timed out (quota may be exhausted)"

        nxt = state.rotate_account(exclude=account)
        if not nxt:
            raise last_error
        typer.echo(
            f"[colab] Account '{account}' was {reason}. Switching to account "
            f"'{nxt}' and retrying.",
            err=True,
        )
    raise last_error


def new(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    tpu: Annotated[
        Optional[str],
        typer.Option(
            help="TPU accelerator variant. Supported: v5e1, v6e1.",
        ),
    ] = None,
    gpu: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "GPU accelerator variant. Supported: T4, L4, G4, H100, A100."
                "\n\nIf omitted (along with --tpu), a CPU runtime is created."
                "\n\nAvailability varies by Colab subscription tier."
            ),
        ),
    ] = None,
    high_mem: Annotated[
        bool,
        typer.Option(
            "--high-mem",
            help=(
                "Request a high-RAM machine shape (CPU, T4, A100, etc.). "
                "Requires Colab Pro or Pro+ entitlement. Ignored for "
                "accelerators that only offer a single shape (L4, v5e1, v6e1)."
            ),
        ),
    ] = False,
    force_new: Annotated[
        bool,
        typer.Option(
            "--force-new",
            help=(
                "Always request a fresh runtime, even when a matching one is "
                "already running (by default the running one is reused)."
            ),
        ),
    ] = False,
):
    """Create a new session"""
    from colab_cli.common import state

    name = session or uuid.uuid4().hex[:6]
    variant, accelerator, shape = resolve_runtime_options(gpu, tpu, high_mem=high_mem)

    if high_mem and accelerator in HIGH_MEM_ONLY_ACCELERATORS:
        typer.echo(
            "[colab] --high-mem ignored: this accelerator only offers one "
            "machine shape.",
            err=True,
        )

    # Reuse a live runtime with the same hardware instead of asking for a
    # second one and getting refused by the account's concurrency limit.
    if not force_new and _adopt_existing_session(name, variant, accelerator):
        return

    typer.echo(f"[colab] Creating session '{name}'...")
    try:
        res = _assign_with_failover(variant, accelerator, shape)
    except TooManyAssignmentsError:
        # The Colab backend returns 412 when it refuses the assignment. This
        # usually means too many active sessions, or a temporary usage or
        # capacity limit for the requested runtime. Translate that to a
        # friendly, actionable message instead of a raw traceback.
        typer.echo(
            "[colab] Allocation refused (precondition failed). This can mean "
            "too many active sessions, or a temporary usage or capacity "
            "limit for the requested runtime. Run `colab stop` to free up a "
            "session, wait and retry, or try a different accelerator.",
            err=True,
        )
        raise typer.Exit(code=1)
    except ColabRequestError as e:
        # The Colab backend returns 400 when the caller is not entitled to the
        # requested accelerator (e.g. no A100 quota). Translate that to a
        # friendly, actionable message instead of a raw traceback. We only
        # interpret it this way when an accelerator was actually requested;
        # otherwise we re-raise so the user sees the real cause.
        if get_status_code(e) == 400 and accelerator != Accelerator.NONE:
            typer.echo(
                f"[colab] Backend rejected accelerator '{accelerator.value}'. "
                "You may not have quota or entitlement for this accelerator on "
                "your account. Try a different one (e.g. --gpu T4) or omit "
                "--gpu/--tpu for a CPU runtime.",
                err=True,
            )
            raise typer.Exit(code=1)
        raise

    if isinstance(res, PostAssignmentResponse):
        token = res.runtime_proxy_info.token
        url = res.runtime_proxy_info.url
        endpoint = res.endpoint
    else:
        token = (
            res.runtime_proxy_info.token
            if hasattr(res, "runtime_proxy_info")
            else getattr(res, "runtime_proxy_token", "")
        )
        url = res.runtime_proxy_info.url if hasattr(res, "runtime_proxy_info") else ""
        endpoint = res.endpoint

    # Importing locally to avoid a top-level circular import via auth.

    s = SessionState(
        name=name,
        token=token,
        url=url,
        endpoint=endpoint,
        token_expires_at=(
            res.runtime_proxy_info.expires_at()
            if hasattr(res, "runtime_proxy_info")
            else None
        ),
        variant=variant.value,
        accelerator=accelerator.value,
        machine_shape=(
            Shape.HIGH_RAM.name if shape == Shape.HIGH_RAM else Shape.STANDARD.name
        ),
        account=_account_label(state),
    )

    state.store.add(s)
    state.history.log_event(
        name,
        "session_created",
        {
            "endpoint": endpoint,
            "variant": variant.value,
            "accelerator": accelerator.value,
            "machine_shape": s.machine_shape,
        },
    )
    typer.echo("[colab] Session READY.")


def restart_kernel(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
):
    """Restart a session's kernel"""
    from colab_cli.common import state

    name = state.resolve_session(session)
    s = state.get_session(name)

    def on_started(kid):
        s.kernel_id = kid
        state.store.add(s)

    def on_sess_started(sid):
        s.session_id = sid
        state.store.add(s)

    runtime = ColabRuntime(
        s.url,
        s.token,
        kernel_id=s.kernel_id,
        session_id=s.session_id,
        on_kernel_started=on_started,
        on_session_started=on_sess_started,
    )

    try:
        runtime.restart()
    finally:
        runtime.stop()


def sessions_command(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit sessions as a JSON array on stdout.",
        ),
    ] = False,
):
    """List all active sessions"""
    from colab_cli.common import state

    sessions, assignments = state.sync_sessions()
    name_by_endpoint = {s.endpoint: s.name for s in sessions.values()}
    if json_output:
        import json as _json

        typer.echo(
            _json.dumps(
                [
                    {
                        "name": name_by_endpoint.get(a.endpoint, "?"),
                        "endpoint": a.endpoint,
                        "accelerator": a.accelerator.value,
                        "variant": a.variant.name,
                        "machine_shape": a.machine_shape.name,
                    }
                    for a in assignments
                ]
            )
        )
        return
    if not assignments:
        typer.echo("[colab] No active sessions found on server.")
        return
    for a in assignments:
        name = name_by_endpoint.get(a.endpoint, "?")
        # `a.variant` is an int-valued AssignmentVariant (DEFAULT=0/GPU=1/TPU=2);
        # its `.name` matches the user-facing string Variant enum, which is what
        # `status` shows for locally-tracked sessions.
        typer.echo(
            _format_session_line(
                name=name,
                endpoint=a.endpoint,
                accelerator=a.accelerator.value,
                variant=a.variant.name,
                machine_shape=a.machine_shape.name,
            )
        )


def _print_status_for(s: SessionState) -> None:
    """Print one session's status line plus optional last-execution detail."""
    status = f"BUSY ({s.running})" if s.running else "IDLE"
    typer.echo(
        _format_session_line(
            name=s.name,
            endpoint=s.endpoint,
            accelerator=s.accelerator,
            variant=s.variant,
            status=status,
            machine_shape=s.machine_shape,
        )
    )
    if s.last_execution:
        exec_file, exec_cell, exec_time = s.last_execution
        cell_str = f" | Cell: {exec_cell}" if exec_cell else ""
        typer.echo(f"  Last Execution: {exec_file}{cell_str} at {exec_time}")


def status(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit session status as JSON on stdout.",
        ),
    ] = False,
):
    """Show session status"""
    from colab_cli.common import state

    def _as_json(s) -> dict:
        return {
            "name": s.name,
            "endpoint": s.endpoint,
            "accelerator": s.accelerator,
            "variant": s.variant,
            "machine_shape": s.machine_shape,
            "status": f"BUSY ({s.running})" if s.running else "IDLE",
            "running": s.running,
            "last_execution": s.last_execution,
            "kernel_id": s.kernel_id,
            "session_id": s.session_id,
        }

    local_sessions, _ = state.sync_sessions()
    if session:
        s = state.store.get(session)
        if s:
            if json_output:
                import json as _json

                typer.echo(_json.dumps(_as_json(s)))
            else:
                _print_status_for(s)
        else:
            typer.echo(f"[colab] Session '{session}' not found.", err=True)
        return

    if not local_sessions:
        typer.echo("[colab] No active sessions.")
        return
    if json_output:
        import json as _json

        typer.echo(
            _json.dumps([_as_json(s) for s in local_sessions.values()])
        )
        return
    for s in local_sessions.values():
        _print_status_for(s)


def stop(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
):
    """Stop a session"""
    from colab_cli.common import state

    name = state.resolve_session(session)
    s = state.get_session(name, ignore_missing_session=True)
    if not s:
        typer.echo(f"[colab] Session '{name}' not found.", err=True)
        return

    typer.echo(f"[colab] Stopping session '{name}'...")
    try:
        runtime = ColabRuntime(s.url, s.token, kernel_id=s.kernel_id)
        runtime.stop(shutdown_kernel=True)
    except Exception:
        pass

    # Unassign under the account that owns the runtime: a session created
    # after a failover lives on a different account and is invisible to the
    # active one.
    owner = s.account if isinstance(s.account, str) and s.account else None
    current = _account_label(state)
    if owner and current and owner != current:
        state.client_for(owner).unassign(s.endpoint)
    else:
        state.client.unassign(s.endpoint)
    state.store.remove(name)
    state.history.log_event(name, "session_terminated", {"reason": "user_requested"})
    typer.echo("[colab] Session terminated.")


def register(app: typer.Typer):
    app.command()(new)
    app.command(name="sessions")(sessions_command)
    app.command(name="restart-kernel")(restart_kernel)
    app.command()(status)
    app.command()(stop)

