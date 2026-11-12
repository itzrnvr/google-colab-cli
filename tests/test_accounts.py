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

"""Multi-account support: the account store and the `colab account` commands."""

import json
import os

import pytest
from typer.testing import CliRunner

from colab_cli.accounts import (
    DEFAULT_ACCOUNT,
    AccountError,
    AccountStore,
    validate_label,
)
from colab_cli.cli import app

runner = CliRunner()


@pytest.fixture
def store(tmp_path):
    """An AccountStore fully isolated in a temp directory."""
    return AccountStore(
        index_path=str(tmp_path / "accounts.json"),
        accounts_dir=str(tmp_path / "accounts"),
        legacy_token_path=str(tmp_path / "token.json"),
    )


def _write_token(path, payload=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload or {"token": "t"}, f)


# --------------------------------------------------------------- the store


def test_default_account_maps_to_legacy_token_path(store):
    """The legacy single-token layout must keep working untouched."""
    assert store.token_path(DEFAULT_ACCOUNT) == store.legacy_token_path
    assert store.token_path("work") == os.path.join(
        store.accounts_dir, "work.json"
    )


def test_validate_label_rejects_unsafe_names():
    assert validate_label("work-2") == "work-2"
    assert validate_label("a.b_c") == "a.b_c"
    for bad in ["", "  ", "-leading", "has space", "sl/ash", "x" * 65, "é"]:
        with pytest.raises(AccountError):
            validate_label(bad)


def test_save_token_registers_account_with_email(store):
    store.save_token("work", {"token": "abc"}, email="me@example.com")
    assert store.has("work")
    assert store.known("work")
    assert store.meta("work")["email"] == "me@example.com"
    with open(store.token_path("work"), encoding="utf-8") as f:
        assert json.load(f) == {"token": "abc"}


def test_active_account_defaults_to_legacy_when_no_index(store):
    """With only the legacy token present, `default` is the active account."""
    assert store.active() is None
    _write_token(store.legacy_token_path)
    assert store.active() == DEFAULT_ACCOUNT


def test_set_active_requires_a_token(store):
    with pytest.raises(AccountError):
        store.set_active("ghost")
    store.save_token("work", {"token": "abc"})
    store.set_active("work")
    assert store.active() == "work"


def test_remove_account_clears_active(store):
    store.save_token("work", {"token": "abc"})
    store.set_active("work")
    store.remove("work")
    assert store.active() is None
    assert not store.has("work")


def test_next_account_skips_the_current_one(store):
    store.save_token("a", {"token": "1"})
    store.save_token("b", {"token": "2"})
    store.save_token("c", {"token": "3"})
    assert store.next_account(exclude="a") in {"b", "c"}
    assert store.next_account(exclude="zzz") in {"a", "b", "c"}


def test_next_account_returns_none_when_alone(store):
    store.save_token("solo", {"token": "1"})
    assert store.next_account(exclude="solo") is None
    assert store.next_account(exclude="other") == "solo"


def test_next_account_ignores_labels_without_tokens(store):
    store.save_token("real", {"token": "1"})
    store.register("ghost")  # indexed but no token on disk
    assert store.usable_accounts() == ["real"]
    assert store.next_account(exclude="real") is None


def test_rotation_advances_instead_of_repeating(store):
    store.save_token("a", {"token": "1"})
    store.save_token("b", {"token": "2"})
    first = store.next_account(exclude="x")
    store.advance_rotation(first)
    second = store.next_account(exclude="x")
    assert second != first


# ------------------------------------------------------------- the commands


@pytest.fixture
def isolated_accounts(tmp_path, monkeypatch):
    """Point the account commands at a temp directory.

    ``AccountStore`` resolves its default paths from the module globals at
    call time, so patching them in ``colab_cli.accounts`` redirects every
    store built by the commands.
    """
    monkeypatch.setattr(
        "colab_cli.accounts.ACCOUNTS_INDEX_PATH", str(tmp_path / "accounts.json")
    )
    monkeypatch.setattr(
        "colab_cli.accounts.ACCOUNTS_DIR", str(tmp_path / "accounts")
    )
    monkeypatch.setattr(
        "colab_cli.accounts.LEGACY_TOKEN_PATH", str(tmp_path / "token.json")
    )
    return tmp_path


def _store_for(tmp_path):
    return AccountStore(
        index_path=str(tmp_path / "accounts.json"),
        accounts_dir=str(tmp_path / "accounts"),
        legacy_token_path=str(tmp_path / "token.json"),
    )


def test_account_list_is_empty_without_accounts(isolated_accounts):
    result = runner.invoke(app, ["account", "list"])
    assert result.exit_code == 0
    assert "No accounts signed in" in result.stderr


def test_account_list_shows_accounts_and_active(isolated_accounts):
    st = _store_for(isolated_accounts)
    st.save_token("work", {"token": "1"}, email="me@example.com")
    st.set_active("work")

    result = runner.invoke(app, ["account", "list"])
    assert result.exit_code == 0
    assert "work" in result.stdout
    assert "me@example.com" in result.stdout


def test_account_list_json_is_machine_readable(isolated_accounts):
    st = _store_for(isolated_accounts)
    st.save_token("work", {"token": "1"}, email="me@example.com")
    st.set_active("work")

    result = runner.invoke(app, ["account", "list", "--json"])
    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc[0]["label"] == "work"
    assert doc[0]["active"] is True
    assert doc[0]["email"] == "me@example.com"


def test_account_use_switches_the_active_account(isolated_accounts):
    st = _store_for(isolated_accounts)
    st.save_token("work", {"token": "1"})
    st.save_token("home", {"token": "2"})

    result = runner.invoke(app, ["account", "use", "home"])
    assert result.exit_code == 0
    # A fresh store: the command wrote the index, this instance cached the
    # pre-command state.
    assert _store_for(isolated_accounts).active() == "home"


def test_account_use_unknown_label_fails_cleanly(isolated_accounts):
    result = runner.invoke(app, ["account", "use", "ghost"])
    assert result.exit_code == 2
    assert "ghost" in result.stderr


def test_account_rm_removes_the_token(isolated_accounts):
    st = _store_for(isolated_accounts)
    st.save_token("work", {"token": "1"})

    result = runner.invoke(app, ["account", "rm", "work"])
    assert result.exit_code == 0
    assert not st.has("work")


def test_account_current_prints_the_label(isolated_accounts):
    st = _store_for(isolated_accounts)
    st.save_token("work", {"token": "1"})
    st.set_active("work")

    result = runner.invoke(app, ["account", "current"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "work"


def test_global_account_option_rejects_bad_labels():
    result = runner.invoke(app, ["--account", "bad label", "account", "list"])
    assert result.exit_code == 2
    assert "Invalid account label" in result.stderr
