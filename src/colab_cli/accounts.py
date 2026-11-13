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

"""Multi-account support for the Colab CLI.

Colab enforces *per-Google-account* limits: a daily compute quota and a
concurrency limit on GPU assignments. When one account's quota is exhausted,
every ``colab-tool new --gpu T4`` fails with HTTP 412 (``Precondition Failed``)
even though other signed-in accounts still have headroom.

This module stores one OAuth token per labelled account so a machine (or an
agent) can hold several accounts and rotate to a healthy one when the active
account is throttled.

Layout under ``~/.config/colab-cli``::

    token.json              legacy single-account token (the "default" account)
    accounts.json           index: active label + per-account metadata
    accounts/<label>.json   OAuth token for a non-default account

The ``default`` label deliberately maps onto the legacy ``token.json`` path so
existing installations keep working untouched.
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_ACCOUNT = "default"

# Path definitions live here (and not in auth.py) so there is exactly one
# spelling of each path. `os.path.expanduser` keeps forward slashes on
# Windows, so building these with `os.path.join` would produce a *different*
# string than the legacy `TOKEN_CONFIG_PATH` and break path comparisons.
LEGACY_TOKEN_PATH = os.path.expanduser("~/.config/colab-cli/token.json")
ACCOUNTS_INDEX_PATH = os.path.expanduser("~/.config/colab-cli/accounts.json")
ACCOUNTS_DIR = os.path.expanduser("~/.config/colab-cli/accounts")

# Label rules: keep them filesystem-safe so a label can be used as a filename
# without any escaping games.
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# Order used when rotating between accounts for failover. Prefer accounts that
# were used least recently, so a burst of failures does not hammer one account.
_ROTATION_STATE_FILENAME = "rotation.json"


class AccountError(Exception):
    """Raised for invalid account operations (unknown label, bad name, ...)."""


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def validate_label(label: str) -> str:
    label = (label or "").strip()
    if not _LABEL_RE.match(label):
        raise AccountError(
            f"Invalid account label {label!r}. Use letters, digits, '.', '_' or "
            "'-' (max 64 chars, must start with a letter or digit)."
        )
    return label


class AccountStore:
    """Reads and writes the multi-account index plus per-account token files."""

    def __init__(
        self,
        index_path: Optional[str] = None,
        accounts_dir: Optional[str] = None,
        legacy_token_path: Optional[str] = None,
    ):
        self.index_path = index_path or ACCOUNTS_INDEX_PATH
        self.accounts_dir = accounts_dir or ACCOUNTS_DIR
        self.legacy_token_path = legacy_token_path or LEGACY_TOKEN_PATH
        self._index: Optional[Dict] = None

    # ------------------------------------------------------------------ io

    def _load_index(self) -> Dict:
        if self._index is not None:
            return self._index
        index = {"version": 1, "active": None, "accounts": {}}
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                index.update(loaded)
                if not isinstance(index.get("accounts"), dict):
                    index["accounts"] = {}
        except FileNotFoundError:
            pass
        except Exception as e:  # pragma: no cover - corrupt index shouldn't be fatal
            logger.warning(f"Ignoring unreadable account index {self.index_path}: {e}")
        self._index = index
        return index

    def _save_index(self) -> None:
        index = self._load_index()
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        tmp = self.index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.index_path)

    # ------------------------------------------------------------- queries

    def labels(self) -> List[str]:
        # The legacy single-token file is the "default" account even when it
        # was never registered in the index (pre-multi-account installs).
        labels = set(self._load_index()["accounts"].keys())
        if os.path.exists(self.legacy_token_path):
            labels.add(DEFAULT_ACCOUNT)
        return sorted(labels)

    def meta(self, label: str) -> Dict:
        return dict(self._load_index()["accounts"].get(label, {}))

    def active(self) -> Optional[str]:
        active = self._load_index().get("active")
        if active and active in self._load_index()["accounts"]:
            return active
        # Fall back to the legacy token file when the index has no active entry
        # but a legacy token exists: that is the pre-multi-account world.
        if os.path.exists(self.legacy_token_path):
            return DEFAULT_ACCOUNT
        return None

    def token_path(self, label: str) -> str:
        """Filesystem path holding the OAuth token for ``label``."""
        label = validate_label(label)
        if label == DEFAULT_ACCOUNT:
            return self.legacy_token_path
        return os.path.join(self.accounts_dir, f"{label}.json")

    def has(self, label: str) -> bool:
        return os.path.exists(self.token_path(label))

    def known(self, label: str) -> bool:
        return label in self._load_index()["accounts"]

    def usable_accounts(self) -> List[str]:
        """Labels that have a token file on disk, ordered by the index."""
        return [label for label in self.labels() if self.has(label)]

    # ------------------------------------------------------------ mutation

    def register(self, label: str, email: Optional[str] = None) -> None:
        label = validate_label(label)
        index = self._load_index()
        meta = index["accounts"].setdefault(label, {})
        if email:
            meta["email"] = email
        meta.setdefault("added_at", _now_iso())
        self._save_index()

    def save_token(self, label: str, token_json: Dict, email: Optional[str] = None) -> str:
        """Persist ``token_json`` under ``label`` and record it in the index."""
        label = validate_label(label)
        path = self.token_path(label)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(token_json, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
        self.register(label, email=email)
        return path

    def set_active(self, label: str) -> None:
        label = validate_label(label)
        if not self.has(label):
            raise AccountError(
                f"Account '{label}' has no stored token. Run "
                f"`colab-tool account add {label}` to sign in first."
            )
        index = self._load_index()
        index["active"] = label
        meta = index["accounts"].setdefault(label, {})
        meta["last_used"] = _now_iso()
        self._save_index()

    def remove(self, label: str) -> None:
        label = validate_label(label)
        index = self._load_index()
        index["accounts"].pop(label, None)
        if index.get("active") == label:
            remaining = self.usable_accounts()
            index["active"] = remaining[0] if remaining else None
        self._save_index()
        path = self.token_path(label)
        if os.path.exists(path):
            os.remove(path)

    def touch(self, label: str) -> None:
        """Record that ``label`` was just used (best effort)."""
        try:
            index = self._load_index()
            meta = index["accounts"].setdefault(label, {})
            meta["last_used"] = _now_iso()
            self._save_index()
        except Exception as e:  # pragma: no cover
            logger.debug(f"Could not record last_used for account {label}: {e}")

    # ------------------------------------------------------------ failover

    def _rotation_state_path(self) -> str:
        return os.path.join(self.accounts_dir, _ROTATION_STATE_FILENAME)

    def _read_rotation_state(self) -> Dict:
        try:
            with open(self._rotation_state_path(), "r", encoding="utf-8") as f:
                state = json.load(f)
            return state if isinstance(state, dict) else {}
        except Exception:
            return {}

    def _write_rotation_state(self, state: Dict) -> None:
        try:
            os.makedirs(self.accounts_dir, exist_ok=True)
            with open(self._rotation_state_path(), "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception as e:  # pragma: no cover
            logger.debug(f"Could not persist rotation state: {e}")

    def next_account(self, exclude: Optional[str] = None) -> Optional[str]:
        """Return the next account to try after ``exclude`` failed.

        Returns ``None`` when there is no other account with a stored token.
        The order is round-robin over the usable accounts, persisted across
        invocations so that consecutive failures keep moving forward instead of
        retrying the same dead account.
        """
        candidates = [label for label in self.usable_accounts() if label != exclude]
        if not candidates:
            return None
        state = self._read_rotation_state()
        cursor = state.get("cursor")
        if cursor in candidates:
            idx = candidates.index(cursor)
            return candidates[(idx + 1) % len(candidates)]
        return candidates[0]

    def advance_rotation(self, label: str) -> None:
        state = self._read_rotation_state()
        state["cursor"] = label
        state["updated_at"] = _now_iso()
        self._write_rotation_state(state)
