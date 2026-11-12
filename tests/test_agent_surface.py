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

"""The agent-facing surface: doctor, the working-directory path resolution,
the --files listing, and stdout purity in --json mode."""

import json

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


def _session():
    return SessionState(
        name="s1", token="tok", url="http://runtime.example", endpoint="e1"
    )


# ------------------------------------------------------- working dir paths


def test_working_dir_variant_maps_both_spellings():
    assert ContentsClient._working_dir_variant("model.json") == "content/model.json"
    assert ContentsClient._working_dir_variant("content/model.json") == "model.json"
    assert ContentsClient._working_dir_variant("content") is None


def test_bare_name_resolves_against_the_working_directory(mocker):
    """`colab download model.json` must find /content/model.json.

    The contents API is rooted at the notebook root while the VM's working
    directory is /content, so a bare name 404s unless it is retried there.
    """
    calls = []

    def fake_request(method, url, params=None, json=None):
        calls.append(url)
        response = mocker.MagicMock()
        if url.endswith("/api/contents/model.json"):
            response.status_code = 404
            response.text = '{"error": {"message": "not found"}}'
        else:
            response.status_code = 200
            response.text = ""
            response.json.return_value = {
                "type": "file",
                "format": "base64",
                "content": "aGVsbG8=",
            }
        response.raise_for_status.return_value = None
        return response

    mocker.patch("colab_cli.contents.requests.request", side_effect=fake_request)

    client = ContentsClient(_session())
    client.download("model.json", "unused")

    assert any(u.endswith("/api/contents/content/model.json") for u in calls), calls


def test_download_of_a_missing_file_explains_the_path(mocker):
    response = mocker.MagicMock()
    response.status_code = 404
    response.text = '{"error": {"message": "nope"}}'
    mocker.patch("colab_cli.contents.requests.request", return_value=response)

    client = ContentsClient(_session())
    with pytest.raises(FileNotFoundError) as exc:
        client.download("model.json", "unused")
    assert "content/model.json" in str(exc.value)


# ---------------------------------------------------------- status --files


def test_status_files_lists_the_remote_artifacts(
    mock_client, mock_store, mock_common_state, mocker
):
    """After a training run the agent must be able to see what to download."""
    sess = mocker.MagicMock()
    sess.name = "s1"
    sess.endpoint = "e1"
    sess.accelerator = "T4"
    sess.variant = "GPU"
    sess.machine_shape = "STANDARD"
    sess.running = None
    sess.last_execution = None
    sess.kernel_id = "k1"
    sess.session_id = "sid1"
    sess.account = "default"
    mock_store.get.return_value = sess
    mock_common_state.sync_sessions.return_value = ({"s1": sess}, [])

    listing = {
        "type": "directory",
        "content": [
            {
                "name": "model.json",
                "path": "content/model.json",
                "type": "file",
                "size": 269,
            },
            {
                "name": "ckpt",
                "path": "content/ckpt",
                "type": "directory",
                "size": None,
            },
        ],
    }
    mocker.patch(
        "colab_cli.contents.ContentsClient.list_dir", return_value=listing
    )

    result = runner.invoke(app, ["status", "-s", "s1", "--json", "--files"])
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["account"] == "default"
    names = [f["name"] for f in doc["files"]]
    assert "model.json" in names
    # The path must be the one that actually works for `colab download`.
    paths = [f["path"] for f in doc["files"]]
    assert "content/model.json" in paths


# -------------------------------------------------------------- doctor


def test_doctor_reports_ready(mocker):
    mocker.patch("colab_cli.commands.doctor.run_checks", return_value=[])
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["ok"] is True
    assert doc["next"] == []


def test_doctor_lists_the_fixes_when_broken(mocker):
    from colab_cli.commands.doctor import Check

    mocker.patch(
        "colab_cli.commands.doctor.run_checks",
        return_value=[
            Check(
                "accounts",
                "fail",
                "no account is signed in",
                "colab account add default",
            ),
        ],
    )
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert doc["ok"] is False
    assert doc["next"] == ["colab account add default"]


def test_doctor_human_output_numbers_the_actions(mocker):
    from colab_cli.commands.doctor import Check

    mocker.patch(
        "colab_cli.commands.doctor.run_checks",
        return_value=[
            Check(
                "accounts",
                "fail",
                "no account is signed in",
                "colab account add default",
            ),
            Check("interpreter", "ok", "fine"),
        ],
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "NOT READY" in result.stderr
    assert "1. colab account add default" in result.stderr
