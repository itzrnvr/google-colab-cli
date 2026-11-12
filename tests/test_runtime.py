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

from unittest.mock import MagicMock, patch

import jupyter_kernel_client

from colab_cli.runtime import ColabRuntime


def test_colab_runtime_kernel_client():
    if hasattr(jupyter_kernel_client, "ColabKernelClient"):
        target_attr = "ColabKernelClient"
        token_param_name = "proxy_token"
    elif hasattr(jupyter_kernel_client, "KernelClient"):
        target_attr = "KernelClient"
        token_param_name = "token"
    else:
        target_attr = "JupyterKernelClient"
        token_param_name = "token"

    with patch.object(jupyter_kernel_client, target_attr) as mock_kc_cls:
        mock_kc = mock_kc_cls.return_value
        runtime = ColabRuntime("http://url", "token123")

        assert runtime._kernel_client is None

        kc = runtime.kernel_client

        expected_kwargs = {
            "server_url": "http://url",
            token_param_name: "token123",
            "kernel_id": None,
            "client_kwargs": {
                "subprotocol": jupyter_kernel_client.JupyterSubprotocol.DEFAULT,
                "extra_params": {"colab-runtime-proxy-token": "token123"},
            },
            "headers": {
                "X-Colab-Client-Agent": "colab-cli",
                "X-Colab-Runtime-Proxy-Token": "token123",
            },
        }
        mock_kc_cls.assert_called_once_with(**expected_kwargs)
        mock_kc.start.assert_called_once()
        assert kc == mock_kc


def test_colab_runtime_kernel_client_jupyterclient_fallback():
    """PyPI jupyter-kernel-client>=1.0 renamed KernelClient to
    JupyterKernelClient. Runtime must fall back to it instead of raising
    AttributeError (breaks `colab exec` on `uv tool install google-colab-cli`)."""
    import colab_cli.runtime as runtime_mod

    real_colab = getattr(jupyter_kernel_client, "ColabKernelClient", None)
    real_legacy = getattr(jupyter_kernel_client, "KernelClient", None)
    had_colab = hasattr(jupyter_kernel_client, "ColabKernelClient")
    had_legacy = hasattr(jupyter_kernel_client, "KernelClient")

    if had_colab:
        delattr(jupyter_kernel_client, "ColabKernelClient")
    if had_legacy:
        delattr(jupyter_kernel_client, "KernelClient")
    try:
        with patch.object(
            jupyter_kernel_client, "JupyterKernelClient", create=True
        ) as mock_kc_cls:
            mock_kc = mock_kc_cls.return_value
            runtime = ColabRuntime("http://url", "token123")
            kc = runtime.kernel_client
            mock_kc_cls.assert_called_once_with(
                server_url="http://url",
                token="token123",
                kernel_id=None,
                client_kwargs={
                    "subprotocol": jupyter_kernel_client.JupyterSubprotocol.DEFAULT,
                    "extra_params": {"colab-runtime-proxy-token": "token123"},
                },
                headers={
                    "X-Colab-Client-Agent": "colab-cli",
                    "X-Colab-Runtime-Proxy-Token": "token123",
                },
            )
            mock_kc.start.assert_called_once()
            assert kc == mock_kc
    finally:
        if had_colab:
            jupyter_kernel_client.ColabKernelClient = real_colab
        if had_legacy:
            jupyter_kernel_client.KernelClient = real_legacy
        assert runtime_mod  # keep import used for clarity


def test_colab_runtime_execute_code():
    runtime = ColabRuntime("http://url", "token123")
    mock_kc = MagicMock()
    runtime._kernel_client = mock_kc

    # Test empty reply
    mock_kc.execute.return_value = {}
    assert runtime.execute_code("print(1)") == []

    # Test normal reply
    mock_kc.execute.return_value = {"outputs": [{"text": "1\n"}]}
    assert runtime.execute_code("print(1)") == [{"text": "1\n"}]

    # Test error status without error output
    mock_kc.execute.return_value = {
        "status": "error",
        "ename": "ValueError",
        "evalue": "bad",
        "outputs": [{"text": "partial"}],
    }
    outputs = runtime.execute_code("raise ValueError")
    assert len(outputs) == 2
    assert outputs[0] == {"text": "partial"}
    assert outputs[1] == {
        "output_type": "error",
        "ename": "ValueError",
        "evalue": "bad",
        "traceback": [],
    }


def test_colab_runtime_execute_code_default_no_timeout():
    """The kernel call must never carry a deadline.

    jupyter-kernel-client turns any timeout into a 100% CPU spin that never
    raises TimeoutError (it recomputes `timeout = max(0, deadline - now)` and
    then polls with `wait(0)` / `get_msg(timeout=0)`). Omitting the argument
    is just as bad: the client then uses its own 10 s default and spins
    after 10 s.

    So the call is made with an explicit `timeout=None` -- which makes the
    client block on the message event instead of spinning -- and the budget
    is enforced by `_run_with_deadline` outside the client.
    """
    runtime = ColabRuntime("http://url", "token123")
    mock_kc = MagicMock()
    runtime._kernel_client = mock_kc

    mock_kc.execute.return_value = {"outputs": []}
    runtime.execute_code("print(1)")

    _, kwargs = mock_kc.execute.call_args
    assert kwargs.get("timeout", "missing") is None


def test_colab_runtime_execute_code_with_timeout():
    """A timeout is enforced by the runtime, not handed to the client."""
    runtime = ColabRuntime("http://url", "token123")
    mock_kc = MagicMock()
    runtime._kernel_client = mock_kc

    mock_kc.execute.return_value = {"outputs": []}
    runtime.execute_code("print(1)", timeout=600)

    _, kwargs = mock_kc.execute.call_args
    assert kwargs.get("timeout") is None


def test_colab_runtime_execute_interactive_with_timeout():
    """Same for the execute_interactive branch (used with an output_hook)."""
    runtime = ColabRuntime("http://url", "token123")
    mock_kc = MagicMock()
    runtime._kernel_client = mock_kc

    mock_kc.execute_interactive.return_value = {"content": {"status": "ok"}}
    runtime.execute_code("print(1)", output_hook=lambda o: None, timeout=600)

    _, kwargs = mock_kc.execute_interactive.call_args
    assert kwargs.get("timeout") is None


def test_colab_runtime_stop():
    runtime = ColabRuntime("http://url", "token123")
    mock_kc = MagicMock()
    runtime._kernel_client = mock_kc

    runtime.stop()
    mock_kc._manager.client.stop_channels.assert_called_once()


def test_colab_runtime_stop_exception(caplog):
    runtime = ColabRuntime("http://url", "token123")
    mock_kc = MagicMock()
    mock_kc._manager.client.stop_channels.side_effect = Exception("Stop failed")
    runtime._kernel_client = mock_kc

    runtime.stop()  # Should not raise
    assert "Error stopping kernel client" in caplog.text


def test_colab_runtime_stdin_logging():
    mock_history = MagicMock()
    runtime = ColabRuntime(
        "http://url", "token", session_name="test-s", history=mock_history
    )
    mock_kc = MagicMock()
    runtime._kernel_client = mock_kc

    mock_kc.execute.side_effect = lambda code, allow_stdin=False, stdin_hook=None, timeout=None: {
        "outputs": [{"text": stdin_hook("Enter something: ")}]
    }

    with patch("colab_cli.runtime.input", return_value="user input"):
        outputs = runtime.execute_code("code", allow_stdin=True)

    assert outputs == [{"text": "user input"}]
    mock_history.log_event.assert_any_call(
        "test-s", "stdin_request", {"prompt": "Enter something: "}
    )
    mock_history.log_event.assert_any_call(
        "test-s", "input_reply", {"value": "user input"}
    )
