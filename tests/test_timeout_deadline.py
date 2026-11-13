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

"""Wall-clock enforcement for kernel execution.

``jupyter_kernel_client`` treats its ``timeout`` as an *idle* budget: once the
deadline passes it recomputes ``timeout = max(0, deadline - now)`` and then
loops with ``wait(0)`` / ``get_msg(timeout=0)``. ``wait(0)`` returns
immediately and ``get_msg(timeout=0)`` raises ``Empty``, so the poll loop
spins at 100% CPU and never raises ``TimeoutError`` -- the remote cell keeps
running (still billing Colab compute quota) and the CLI never returns.

``ColabRuntime._run_with_deadline`` enforces the deadline from the outside.
"""

import threading
import time

import pytest

from colab_cli.runtime import ColabRuntime, ExecutionTimeout


def _runtime():
    return ColabRuntime(url="http://kernel", token="tok")


def test_deadline_returns_value_when_the_call_finishes_in_time():
    rt = _runtime()
    assert rt._run_with_deadline(lambda: "done", 5.0, "Code execution") == "done"


def test_deadline_is_none_runs_inline():
    rt = _runtime()
    assert rt._run_with_deadline(lambda: 42, None, "Code execution") == 42


def test_deadline_reraises_the_original_exception():
    rt = _runtime()

    def boom():
        raise ValueError("kernel exploded")

    with pytest.raises(ValueError, match="kernel exploded"):
        rt._run_with_deadline(boom, 5.0, "Code execution")


def test_deadline_raises_execution_timeout_instead_of_spinning():
    """A call that never returns must time out, not spin forever."""
    rt = _runtime()
    release = threading.Event()

    def hang():
        release.wait(30)
        return "never"

    try:
        start = time.monotonic()
        with pytest.raises(ExecutionTimeout) as exc:
            rt._run_with_deadline(hang, 0.5, "Code execution")
        elapsed = time.monotonic() - start
    finally:
        release.set()

    assert elapsed < 10, "the deadline must fire promptly, not spin"
    assert "exceeded the 0.5s timeout" in str(exc.value)


def test_deadline_interrupts_the_kernel_on_timeout():
    """The kernel must be interrupted, otherwise the cell keeps running."""
    rt = _runtime()
    interrupted = threading.Event()

    class FakeClient:
        def interrupt(self):
            interrupted.set()

    rt._kernel_client = FakeClient()
    release = threading.Event()

    try:
        with pytest.raises(ExecutionTimeout):
            rt._run_with_deadline(lambda: release.wait(30), 0.5, "Code execution")
    finally:
        release.set()

    assert interrupted.is_set(), "the kernel must be interrupted on timeout"


def test_deadline_message_names_the_session_teardown():
    rt = _runtime()
    release = threading.Event()
    try:
        with pytest.raises(ExecutionTimeout) as exc:
            rt._run_with_deadline(lambda: release.wait(30), 0.5, "Code execution")
    finally:
        release.set()
    assert "colab-tool stop" in str(exc.value)


def test_execute_code_uses_the_deadline(mocker):
    """`execute_code` must route the kernel call through the guard."""
    rt = _runtime()
    rt._kernel_client = mocker.MagicMock()

    # A kernel call that outlives the budget must raise, not hang.
    def hang(*args, **kwargs):
        time.sleep(30)

    rt._kernel_client.execute.side_effect = hang
    rt._kernel_client.execute_interactive.side_effect = hang

    with pytest.raises(ExecutionTimeout):
        rt.execute_code("print(1)", timeout=0.5)
