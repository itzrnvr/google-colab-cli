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

"""`colab-tool exec -f <script>` on a script too big to ship inline.

Observed on a live Colab T4: a 246 KB script whose payload was one
243 KB line arrived on the VM corrupted, and the VM raised
"Expecting value: line 1 column 1" while parsing its embedded JSON.
The inline transport mangles very long single lines.

So a script that is too big, or has a line that is too long, is uploaded to
the VM and run from there instead.
"""

import os

from colab_cli.commands.execution import (
    MAX_INLINE_SCRIPT_LINE,
    MAX_INLINE_SCRIPT_BYTES,
    _read_script_for_exec,
    _script_needs_file_transfer,
)


def test_small_script_is_sent_inline(tmp_path):
    p = tmp_path / "small.py"
    p.write_text("print('hello')\n", encoding="utf-8")
    assert not _script_needs_file_transfer(str(p))
    assert _read_script_for_exec(str(p), session_state=None) == "print('hello')\n"


def test_a_long_line_needs_a_file_transfer(tmp_path):
    p = tmp_path / "long.py"
    p.write_text("x = '" + "a" * (MAX_INLINE_SCRIPT_LINE + 10) + "'\n", encoding="utf-8")
    assert _script_needs_file_transfer(str(p))


def test_a_big_file_needs_a_file_transfer(tmp_path):
    p = tmp_path / "big.py"
    p.write_text("x = 1\n" * (MAX_INLINE_SCRIPT_BYTES // 7 + 10), encoding="utf-8")
    assert _script_needs_file_transfer(str(p))


def test_a_big_script_is_uploaded_and_run_from_disk(tmp_path, mocker):
    p = tmp_path / "big.py"
    p.write_text("x = '" + "a" * (MAX_INLINE_SCRIPT_LINE + 10) + "'\n", encoding="utf-8")

    upload = mocker.patch("colab_cli.contents.ContentsClient.upload")
    client = mocker.patch("colab_cli.contents.ContentsClient")
    client.return_value.upload = upload

    code = _read_script_for_exec(str(p), session_state="SESSION")

    # It is uploaded to the VM...
    assert upload.called
    assert upload.call_args[0][0] == str(p)
    assert upload.call_args[0][1] == "/content/_colab_exec_script.py"
    # ...and the code sent is a loader, not the script.
    assert "runpy.run_path('/content/_colab_exec_script.py'" in code
    assert "a" * 100 not in code


def test_the_uploaded_script_name_is_stable(tmp_path, mocker):
    p = tmp_path / "big.py"
    p.write_text("x = '" + "a" * (MAX_INLINE_SCRIPT_LINE + 10) + "'\n", encoding="utf-8")
    client = mocker.patch("colab_cli.contents.ContentsClient")
    _read_script_for_exec(str(p), session_state="S")
    upload = client.return_value.upload
    assert upload.call_args[0][1].endswith("_colab_exec_script.py")
    assert os.path.basename(upload.call_args[0][1]).startswith("_colab_exec_script")
