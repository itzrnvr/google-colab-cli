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

"""The mistakes an agent actually makes, and the errors it should get back.

Every case here was observed in the OMP session transcripts from October
2026, where agents lost time to misleading errors rather than to the real
problem.
"""

import os

import pytest
from typer.testing import CliRunner

from colab_cli.cli import app
from colab_cli.contents import ContentsClient
from colab_cli.state import SessionState

runner = CliRunner()


@pytest.fixture
def mock_client(mock_common_state):
    return mock_common_state.client


@pytest.fixture
def mock_store(mock_common_state):
    return mock_common_state.store


# ------------------------------------------------- `exec -f` with a remote path


def test_exec_with_a_remote_path_explains_that_f_is_local(
    mock_client, mock_store, mock_common_state
):
    """`-f /content/x.py` used to raise a bare FileNotFoundError.

    An agent reads that as "my file is missing" and re-uploads it forever.
    """
    mock_common_state.resolve_session.return_value = "s1"

    result = runner.invoke(app, ["exec", "-s", "s1", "-f", "/content/train.py"])
    assert result.exit_code == 2
    assert "does not exist on this machine" in result.stderr
    assert "LOCALLY" in result.stderr
    # It must point at the working alternative.
    assert "cat /content/train.py | colab-tool exec" in result.stderr


def test_exec_with_a_missing_local_path_still_reports_the_path(
    mock_client, mock_store, mock_common_state
):
    """A path that does not look remote keeps the plain error."""
    mock_common_state.resolve_session.return_value = "s1"

    result = runner.invoke(app, ["exec", "-s", "s1", "-f", "missing.py"])
    assert result.exit_code == 2
    assert "missing.py" in result.stderr
    assert "does not exist on this machine" in result.stderr


# ---------------------------------------------------------- 404 disambiguation


def _session():
    return SessionState(
        name="s1",
        token="tok",
        url="http://runtime.example",
        endpoint="e1",
    )


def test_contents_404_mentions_both_causes(mocker):
    """A dead runtime must not look like a missing file."""
    response = mocker.MagicMock()
    response.status_code = 404
    response.text = '{"error": {"code": 404, "message": "Not Found"}}'
    mocker.patch("colab_cli.contents.requests.request", return_value=response)

    client = ContentsClient(_session())
    with pytest.raises(FileNotFoundError) as exc:
        client.list_dir("content")

    message = str(exc.value)
    assert "File or directory not found: content" in message
    # The agent needs to know the runtime may be gone, not just the path.
    assert "runtime" in message
    assert "colab-tool status" in message


def test_contents_404_includes_the_backend_detail(mocker):
    response = mocker.MagicMock()
    response.status_code = 404
    response.text = '{"error": {"message": "kernel died"}}'
    mocker.patch("colab_cli.contents.requests.request", return_value=response)

    client = ContentsClient(_session())
    with pytest.raises(FileNotFoundError) as exc:
        client.list_dir("content")
    assert "kernel died" in str(exc.value)
