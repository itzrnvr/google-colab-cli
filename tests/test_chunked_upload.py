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

"""Chunked upload: the API appends each chunk, and the size is verified.

A single request carrying the whole file is aborted by the host well before
100 MB (measured: 50 MB works, 100 MB dies with ConnectionAbortedError
10053), so big files are split. A split that silently loses bytes is worse
than a loud failure, so the remote size is checked afterwards.
"""

import base64

import pytest

from colab_cli.contents import UPLOAD_CHUNK_BYTES, ContentsClient
from colab_cli.state import SessionState


@pytest.fixture
def client():
    return ContentsClient(
        SessionState(
            name="s1", token="tok", url="http://runtime.example", endpoint="e1"
        )
    )


def _fake_request(mocker, seen, remote_size=None):
    def _req(method, url, params=None, json=None, **kw):
        if method == "PUT":
            seen.append(json)
        else:
            resp = mocker.MagicMock()
            resp.json.return_value = {"size": remote_size}
            return resp
        resp = mocker.MagicMock()
        resp.json.return_value = {"size": remote_size}
        resp.text = "{}"
        resp.status_code = 200
        return resp

    return mocker.patch("colab_cli.contents.requests.request", side_effect=_req)


def test_small_file_goes_up_in_one_request(client, mocker, tmp_path):
    seen = []
    _fake_request(mocker, seen, remote_size=10)
    src = tmp_path / "small.bin"
    src.write_bytes(b"0123456789")

    client.upload(str(src), "/content/small.bin")

    assert len(seen) == 1
    assert seen[0]["chunk"] == 1
    assert base64.b64decode(seen[0]["content"]) == b"0123456789"


def test_large_file_is_split_into_ordered_chunks(client, mocker, tmp_path):
    seen = []
    size = UPLOAD_CHUNK_BYTES + 1000
    _fake_request(mocker, seen, remote_size=size)
    src = tmp_path / "big.bin"
    src.write_bytes(b"x" * size)

    client.upload(str(src), "/content/big.bin")

    assert len(seen) > 1, "a big file must be split"
    # Chunk numbers are 1-based and consecutive: the API appends in order.
    assert [c["chunk"] for c in seen] == list(range(1, len(seen) + 1))
    # And the bytes, concatenated, are the file.
    joined = b"".join(base64.b64decode(c["content"]) for c in seen)
    assert len(joined) == size


def test_every_chunk_encodes_on_its_own(client, mocker, tmp_path):
    """No padding in the middle of the stream: each chunk is a multiple of 3."""
    seen = []
    size = UPLOAD_CHUNK_BYTES * 2 + 7
    _fake_request(mocker, seen, remote_size=size)
    src = tmp_path / "big.bin"
    src.write_bytes(b"y" * size)

    client.upload(str(src), "/content/big.bin")

    for c in seen[:-1]:
        assert len(base64.b64decode(c["content"])) % 3 == 0


def test_truncated_upload_fails_loudly(client, mocker, tmp_path):
    """A short file must never look like a successful upload."""
    seen = []
    size = UPLOAD_CHUNK_BYTES + 1000
    _fake_request(mocker, seen, remote_size=size // 2)  # the VM got half
    src = tmp_path / "big.bin"
    src.write_bytes(b"z" * size)

    with pytest.raises(IOError, match="incomplete"):
        client.upload(str(src), "/content/big.bin")


def test_upload_of_a_missing_size_check_still_succeeds(client, mocker, tmp_path):
    """When the API does not report a size, do not fail the upload."""
    seen = []
    _fake_request(mocker, seen, remote_size=None)
    src = tmp_path / "small.bin"
    src.write_bytes(b"0123456789")

    client.upload(str(src), "/content/small.bin")
    assert len(seen) == 1
