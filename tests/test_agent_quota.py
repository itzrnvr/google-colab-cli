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

"""`colab new` must not waste quota.

Two failure modes are covered:

1. Requesting a second GPU runtime while one is already running is refused by
   the account's concurrency limit (HTTP 412). The CLI now adopts the live
   runtime instead of asking for another one.
2. When an account's quota is exhausted the CLI rotates to another signed-in
   account and retries, so the work keeps moving.
"""

from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from colab_cli.cli import app
from colab_cli.client import (
    Accelerator,
    AssignmentVariant,
    ListedAssignment,
    RuntimeProxyInfo,
    Shape,
    TooManyAssignmentsError,
)

runner = CliRunner()


@pytest.fixture
def mock_client(mock_common_state):
    return mock_common_state.client


@pytest.fixture
def mock_store(mock_common_state):
    return mock_common_state.store


@pytest.fixture
def mock_state(mock_common_state):
    """The patched ``colab_cli.common.state`` singleton."""
    return mock_common_state


def _listed(accelerator=Accelerator.T4, endpoint="e-live"):
    return ListedAssignment(
        accelerator=accelerator,
        endpoint=endpoint,
        variant=AssignmentVariant.GPU,
        machineShape=Shape.STANDARD,
        runtimeProxyInfo=RuntimeProxyInfo(
            token="live-token", tokenExpiresInSeconds=3600, url="http://live"
        ),
    )


def test_new_adopts_a_running_runtime_instead_of_allocating_again(
    mock_client, mock_store
):
    """A second T4 request would be refused (412); reuse the live one."""
    mock_client.list_assignments.return_value = [_listed()]
    mock_store.list.return_value = {}

    result = runner.invoke(app, ["new", "-s", "reuse-me", "--gpu", "T4"])
    assert result.exit_code == 0, result.output
    mock_client.assign.assert_not_called()
    assert "Adopted" in result.stdout
    assert "Session READY" in result.stdout

    added = mock_store.add.call_args[0][0]
    assert added.name == "reuse-me"
    assert added.endpoint == "e-live"
    assert added.accelerator == "T4"


def test_new_reuses_the_existing_local_session_for_a_live_runtime(
    mock_client, mock_store
):
    """The runtime is already tracked locally: report it, allocate nothing."""
    mock_client.list_assignments.return_value = [_listed()]

    existing = MagicMock()
    existing.name = "already-named"
    existing.endpoint = "e-live"
    mock_store.list.return_value = {"already-named": existing}

    result = runner.invoke(app, ["new", "-s", "other-name", "--gpu", "T4"])
    assert result.exit_code == 0, result.output
    mock_client.assign.assert_not_called()
    assert "already-named" in result.stderr


def test_force_new_skips_the_reuse(mock_client, mock_store):
    """--force-new must allocate even though a matching runtime is live."""
    mock_client.list_assignments.return_value = [_listed()]
    mock_store.list.return_value = {}

    ok_res = MagicMock()
    ok_res.__class__ = MagicMock
    ok_res.runtime_proxy_info = RuntimeProxyInfo(
        token="t", tokenExpiresInSeconds=3600, url="u"
    )
    ok_res.endpoint = "e-new"
    mock_client.assign.return_value = ok_res

    result = runner.invoke(
        app, ["new", "-s", "fresh", "--gpu", "T4", "--force-new"]
    )
    assert result.exit_code == 0, result.output
    mock_client.assign.assert_called_once()


def test_new_allocates_when_no_matching_runtime_exists(
    mock_client, mock_store
):
    mock_client.list_assignments.return_value = [
        _listed(accelerator=Accelerator.L4, endpoint="e-l4")
    ]
    mock_store.list.return_value = {}

    mock_res = MagicMock()
    mock_res.__class__ = MagicMock
    mock_res.runtime_proxy_info = RuntimeProxyInfo(
        token="t", tokenExpiresInSeconds=3600, url="u"
    )
    mock_res.endpoint = "e-new"
    mock_client.assign.return_value = mock_res

    result = runner.invoke(app, ["new", "-s", "s", "--gpu", "T4"])
    assert result.exit_code == 0, result.output
    mock_client.assign.assert_called_once()


def test_new_rotates_to_another_account_when_refused(
    mock_client, mock_store, mock_state
):
    """HTTP 412 on the active account -> switch account and retry."""
    mock_client.list_assignments.return_value = []
    mock_store.list.return_value = {}

    ok_res = MagicMock()
    ok_res.__class__ = MagicMock
    ok_res.runtime_proxy_info = RuntimeProxyInfo(
        token="t2", tokenExpiresInSeconds=3600, url="u2"
    )
    ok_res.endpoint = "e2"

    calls = []

    def assign(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise TooManyAssignmentsError("Precondition Failed")
        return ok_res

    mock_client.assign.side_effect = assign

    # The state double reports the active account and the rotation target.
    mock_state.current_account.return_value = "acct-a"
    mock_state.rotate_account.return_value = "acct-b"

    result = runner.invoke(app, ["new", "-s", "s", "--gpu", "T4"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 2, "the allocation must be retried on the other account"
    assert "Switching to account 'acct-b'" in result.stderr


def test_new_reports_the_refusal_when_there_is_no_other_account(
    mock_client, mock_store, mock_state
):
    mock_client.list_assignments.return_value = []
    mock_store.list.return_value = {}
    mock_client.assign.side_effect = TooManyAssignmentsError("Precondition Failed")

    mock_state.current_account.return_value = "acct-a"
    mock_state.rotate_account.return_value = None

    result = runner.invoke(app, ["new", "-s", "s", "--gpu", "T4"])
    assert result.exit_code == 1
    assert "Allocation refused" in result.stderr


def test_new_rotates_when_the_allocation_request_times_out(
    mock_client, mock_store, mock_state
):
    """A burned account can hang until the HTTP read timeout expires.

    Observed in practice: the exhausted account times out after 120 s while a
    healthy account answers in seconds. That must rotate too, otherwise the
    agent sits on a dead account.
    """
    import requests

    mock_client.list_assignments.return_value = []
    mock_store.list.return_value = {}

    ok_res = MagicMock()
    ok_res.__class__ = MagicMock
    ok_res.runtime_proxy_info = RuntimeProxyInfo(
        token="t2", tokenExpiresInSeconds=3600, url="u2"
    )
    ok_res.endpoint = "e2"

    calls = []

    def assign(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.exceptions.ReadTimeout("read timeout=120")
        return ok_res

    mock_client.assign.side_effect = assign
    mock_state.current_account.return_value = "acct-a"
    mock_state.rotate_account.return_value = "acct-b"

    result = runner.invoke(app, ["new", "-s", "s", "--gpu", "T4"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 2, "a timed-out allocation must be retried on the other account"
    assert "Switching to account 'acct-b'" in result.stderr
    assert "timed out" in result.stderr


def test_new_records_the_account_that_owns_the_session(
    mock_client, mock_store, mock_state
):
    mock_client.list_assignments.return_value = []
    mock_store.list.return_value = {}

    ok_res = MagicMock()
    ok_res.__class__ = MagicMock
    ok_res.runtime_proxy_info = RuntimeProxyInfo(
        token="t", tokenExpiresInSeconds=3600, url="u"
    )
    ok_res.endpoint = "e"
    mock_client.assign.return_value = ok_res

    mock_state.current_account.return_value = "acct-b"

    result = runner.invoke(app, ["new", "-s", "s", "--gpu", "T4"])
    assert result.exit_code == 0, result.output
    added = mock_store.add.call_args[0][0]
    assert added.account == "acct-b"


def test_stop_unassigns_under_the_account_that_owns_the_session(
    mock_client, mock_store, mock_state
):
    """A session created after a failover lives on a different account."""
    sess = MagicMock()
    sess.endpoint = "e1"
    sess.name = "s1"
    sess.url = "http://url"
    sess.token = "token"
    sess.kernel_id = None
    sess.account = "acct-b"
    mock_store.get.return_value = sess
    mock_state.get_session.return_value = sess

    mock_state.current_account.return_value = "acct-a"
    mock_state.client_for.return_value = MagicMock()

    result = runner.invoke(app, ["stop", "-s", "s1"])
    assert result.exit_code == 0, result.output
    mock_state.client_for.assert_called_once_with("acct-b")
    mock_state.client_for.return_value.unassign.assert_called_once_with("e1")
