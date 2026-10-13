# Copyright 2026 Gaurang Patel
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

"""Windows-port + agent-interface regression tests.

Covers the behaviors this fork adds over upstream:
- `colab exec --json` result envelope on stdout, chatter on stderr.
- `colab exec --stream-jsonl` per-output streaming (instant failure signal).
- `colab exec` exits non-zero when a cell emits an error output.
- `colab status --json` / `colab sessions --json` machine output.
- History log filenames sanitized for Windows-reserved characters.
- ProxyCommand uses CreateProcess quoting on win32.
- Console backend selector honors the platform.
- uv-tool install detection works with Windows path separators.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from colab_cli.cli import app

runner = CliRunner()


def _mock_session(store, runtime_class, code_outputs):
    sess = MagicMock()
    sess.name = "s1"
    sess.url = "http://url"
    sess.token = "token"
    sess.kernel_id = None
    sess.session_id = None
    store.get.return_value = sess
    rt = runtime_class.return_value

    def fake_execute(code, output_hook=None, **kwargs):
        outs = list(code_outputs)
        if output_hook:
            for o in outs:
                output_hook(o)
        return outs

    rt.execute_code.side_effect = fake_execute
    return sess


def test_exec_json_envelope_ok(mock_runtime_class, mock_store, mock_common_state):
    """--json emits a parseable result envelope; chatter leaves stdout."""
    mock_common_state.resolve_session.return_value = "s1"
    _mock_session(
        mock_store,
        mock_runtime_class,
        [{"output_type": "stream", "name": "stdout", "text": "hi\n"}],
    )
    result = runner.invoke(app, ["exec", "-s", "s1", "--json"], input="print('hi')")
    assert result.exit_code == 0, result.output
    # Kernel stream text renders human-readable; the FINAL line is the envelope.
    envelope = json.loads(result.stdout.strip().splitlines()[-1])
    assert envelope["event"] == "result"
    assert envelope["ok"] is True
    assert envelope["exit_code"] == 0
    # Human chatter must not pollute stdout in machine mode.
    assert "[colab]" not in result.stdout


def test_exec_json_envelope_failure(mock_runtime_class, mock_store, mock_common_state):
    """A cell error fails the command AND appears in the envelope."""
    mock_common_state.resolve_session.return_value = "s1"
    _mock_session(
        mock_store,
        mock_runtime_class,
        [{"output_type": "error", "ename": "ValueError", "evalue": "bad"}],
    )
    result = runner.invoke(app, ["exec", "-s", "s1", "--json"], input="raise ValueError")
    assert result.exit_code == 1
    envelope = json.loads(result.stdout.strip().splitlines()[-1])
    assert envelope["event"] == "result"
    assert envelope["ok"] is False
    assert envelope["cells"][0]["ok"] is False


def test_exec_stream_jsonl_instant_error(
    mock_runtime_class, mock_store, mock_common_state
):
    """--stream-jsonl surfaces each output (incl. errors) as it arrives."""
    mock_common_state.resolve_session.return_value = "s1"
    outputs = [
        {"output_type": "stream", "name": "stdout", "text": "step1\n"},
        {"output_type": "error", "ename": "RuntimeError", "evalue": "boom"},
    ]
    _mock_session(mock_store, mock_runtime_class, outputs)
    result = runner.invoke(
        app, ["exec", "-s", "s1", "--stream-jsonl"], input="code()"
    )
    assert result.exit_code == 1
    events = [json.loads(line) for line in result.stdout.strip().splitlines()]
    streamed = [e for e in events if e["event"] == "output"]
    assert len(streamed) == 2
    assert streamed[1]["output"]["ename"] == "RuntimeError"
    assert events[-1]["event"] == "result"


def test_exec_plain_failure_exit_code(
    mock_runtime_class, mock_store, mock_common_state
):
    """Even without --json, a failed cell exits non-zero (wrapper contract)."""
    mock_common_state.resolve_session.return_value = "s1"
    _mock_session(
        mock_store,
        mock_runtime_class,
        [{"output_type": "error", "ename": "NameError", "evalue": "x"}],
    )
    result = runner.invoke(app, ["exec", "-s", "s1"], input="x")
    assert result.exit_code == 1


def test_status_json_shape(mock_common_state):
    """status --json emits the documented machine fields."""
    from colab_cli.commands import session as session_mod

    sess = MagicMock()
    sess.name = "s1"
    sess.endpoint = "ep-1"
    sess.accelerator = "T4"
    sess.variant = "GPU"
    sess.machine_shape = "STANDARD"
    sess.running = None
    sess.last_execution = ["f.py", None, "2026-01-01 00:00:00"]
    sess.kernel_id = "k1"
    sess.session_id = "sid1"
    mock_common_state.sync_sessions.return_value = ({"s1": sess}, [])
    mock_common_state.store.get.return_value = sess
    result = runner.invoke(app, ["status", "-s", "s1", "--json"])
    assert result.exit_code == 0, result.output
    doc = json.loads(result.output)
    assert doc["name"] == "s1"
    assert doc["status"] == "IDLE"
    assert doc["kernel_id"] == "k1"


def test_history_sanitizes_windows_reserved_chars(tmp_path):
    """Session names with : * ? etc. must not break history logging."""
    from colab_cli.history import HistoryLogger, _safe_log_stem

    assert _safe_log_stem('a:b/c\\d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"
    logger = HistoryLogger(log_dir=str(tmp_path))
    logger.log_event('sess:ion/2', "execution", {"code": "1"})
    assert len(list(tmp_path.glob("*.jsonl"))) == 1
    assert logger.get_history('sess:ion/2')[0]["code"] == "1"


def test_proxy_command_win32_quoting(monkeypatch):
    """On win32 the ProxyCommand must survive CreateProcess parsing."""
    import sys as _sys

    from colab_cli.commands import ssh as ssh_module

    monkeypatch.setattr(_sys, "platform", "win32")
    sess = MagicMock()
    sess.name = "my session"
    cmd = ssh_module._proxy_command(sess, None)
    if _sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        shell32 = ctypes.windll.shell32
        shell32.CommandLineToArgvW.argtypes = (
            wintypes.LPCWSTR,
            ctypes.POINTER(ctypes.c_int),
        )
        shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
        argc = ctypes.c_int()
        argv_p = shell32.CommandLineToArgvW(cmd, ctypes.byref(argc))
        argv = [argv_p[i] for i in range(argc.value)]
        ctypes.windll.kernel32.LocalFree(argv_p)
        assert argv[-2:] == ["-s", "my session"]
    else:
        import shlex

        assert shlex.split(cmd)[-2:] == ["-s", "my session"]


def test_console_backend_selector():
    """msvcrt backend only on win32 TTYs; never for pipes."""
    import colab_cli.console as console_mod

    orig = sys.platform
    try:
        console_mod.sys.platform = "win32"
        assert console_mod._use_msvcrt(True) is True
        assert console_mod._use_msvcrt(False) is False
        console_mod.sys.platform = "linux"
        assert console_mod._use_msvcrt(True) is False
    finally:
        console_mod.sys.platform = orig


def test_uv_tool_detect_windows_path(monkeypatch):
    """%APPDATA%\\uv\\tools\\... must resolve to the uv install command."""
    from colab_cli import auto_update

    monkeypatch.setattr(
        sys,
        "executable",
        "C:\\Users\\x\\AppData\\Roaming\\uv\\tools\\google-colab-cli\\Scripts\\python.exe",
    )
    monkeypatch.setattr(
        auto_update, "is_self_install_supported", lambda: True
    )
    assert auto_update._get_install_command() == (
        "uv tool install -U google-colab-cli"
    )
    monkeypatch.setattr(sys, "executable", "C:\\Program Files\\Python312\\python.exe")
    assert auto_update._get_install_command() == (
        "pip install --upgrade google-colab-cli"
    )
