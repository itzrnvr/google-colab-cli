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

import base64
import logging
import os
from typing import Optional
from urllib.parse import quote

import requests

from colab_cli.state import SessionState

logger = logging.getLogger(__name__)

# The VM's working directory. The contents API is rooted at the notebook
# root, so files the user cares about live under this prefix.
WORKING_DIR = "content"

# Files larger than this are uploaded in chunks. A single request carrying the
# whole file is aborted by the host well before 100 MB (measured: 50 MB
# succeeds, 100 MB dies with ConnectionAbortedError 10053), so the chunk size
# is what the API is willing to accept in one body.
UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024


class ContentsClient:
    def __init__(self, session_state: SessionState):
        self.base_url = session_state.url.rstrip("/")
        self.token = session_state.token

    def _request(
        self, method: str, path: str, params: dict = None, json_data: dict = None
    ):
        req_params = {"authuser": "0", "colab-runtime-proxy-token": self.token}
        if params:
            req_params.update(params)

        def _send(target: str):
            # Quote the path, but don't encode slashes so directory paths
            # stay intact
            quoted_path = quote(target.strip("/"), safe="/")
            url = f"{self.base_url}/api/contents/{quoted_path}"
            return requests.request(
                method, url, params=req_params, json=json_data
            )

        response = _send(path)

        if response.status_code == 404:
            # The VM's working directory is /content, but the contents API is
            # rooted at the notebook root, so a bare name ("model.json") does
            # not resolve while "content/model.json" does. Agents hit this
            # constantly, so retry against the working directory before
            # declaring the path wrong.
            alt = self._working_dir_variant(path)
            if alt:
                alt_response = _send(alt)
                if alt_response.status_code != 404:
                    response = alt_response

        if response.status_code == 404:
            # A 404 from the contents API is ambiguous: it can mean the path
            # is wrong, or it can mean the *runtime is gone* (the proxy host
            # stops resolving a reclaimed VM). Reporting only "file not found"
            # sends an agent round in circles re-uploading a file that was
            # never the problem -- and, worse, it hides the fact that the
            # compute quota is still being burned by the dead runtime.
            detail = ""
            try:
                body = response.text or ""
                if body:
                    detail = f" ({body[:200].strip()})"
            except Exception:  # pragma: no cover
                pass
            raise FileNotFoundError(
                f"File or directory not found: {path}{detail}. Either the "
                "path is wrong (remote paths are relative to the notebook "
                "root, e.g. 'content/model.json' rather than 'model.json') or "
                "the runtime is gone -- check `colab-tool status -s <session>` and "
                "`colab-tool sessions`."
            )

        response.raise_for_status()

        # DELETE doesn't return JSON
        if method == "DELETE":
            return None

        return response.json()

    @staticmethod
    def _working_dir_variant(path: str) -> Optional[str]:
        """The other spelling of ``path`` relative to the working directory.

        The contents API is rooted at the notebook root but the VM's working
        directory is ``/content``, so ``model.json`` and ``content/model.json``
        name the same file. Return the counterpart, or None when there is no
        sensible alternative.
        """
        clean = path.strip("/")
        if not clean:
            return None
        if clean == WORKING_DIR:
            return None
        if clean.startswith(WORKING_DIR + "/"):
            # content/x -> x
            return clean[len(WORKING_DIR) + 1 :]
        # x -> content/x
        return f"{WORKING_DIR}/{clean}"

    def list_dir(self, path: str):
        return self._request("GET", path)

    def upload(self, local_path: str, remote_path: str):
        """Upload a file to the VM.

        Files at or below ``UPLOAD_CHUNK_BYTES`` go up in one request.
        Larger ones are split: the contents API appends each chunk to the
        file, so a big upload no longer dies on a single oversized request.

        Measured on this box: a single 100 MB base64 PUT is aborted by the
        host (``ConnectionAbortedError 10053``) while 50 MB succeeds;
        chunked, the same file arrives byte-for-byte.

        The size is verified afterwards, so a server that ever truncates
        fails loudly instead of silently corrupting the file.
        """
        size = os.path.getsize(local_path)
        filename = remote_path.split("/")[-1]

        if size <= UPLOAD_CHUNK_BYTES:
            with open(local_path, "rb") as f:
                content = f.read()

            b64_content = base64.b64encode(content).decode("ascii")

            payload = {
                "name": filename,
                "path": remote_path,
                "type": "file",
                "format": "base64",
                "content": b64_content,
                "chunk": 1,
            }

            return self._request("PUT", remote_path, json_data=payload)

        # Split on a multiple of 3 so every chunk base64-encodes on its own
        # (no padding in the middle of the stream).
        step = (UPLOAD_CHUNK_BYTES // 3) * 3
        chunk_no = 0
        with open(local_path, "rb") as f:
            while True:
                raw = f.read(step)
                if not raw:
                    break
                chunk_no += 1
                payload = {
                    "name": filename,
                    "path": remote_path,
                    "type": "file",
                    "format": "base64",
                    "content": base64.b64encode(raw).decode("ascii"),
                    "chunk": chunk_no,
                }
                self._request("PUT", remote_path, json_data=payload)

        # Verify: never let a truncated upload look like a successful one.
        remote = self._request("GET", remote_path, params={"content": "0"})
        remote_size = remote.get("size") if isinstance(remote, dict) else None
        if remote_size is not None and int(remote_size) != size:
            raise IOError(
                f"Upload of '{local_path}' is incomplete: the VM has "
                f"{remote_size} bytes, expected {size}. The file was "
                "truncated in transit -- do not trust it."
            )
        logger.info(
            "Uploaded %s in %d chunk(s) of up to %d MB",
            local_path,
            chunk_no,
            UPLOAD_CHUNK_BYTES // (1024 * 1024),
        )
        return None

    def download(self, remote_path: str, local_path: str):
        data = self._request("GET", remote_path, params={"content": "1"})

        if data.get("type") == "directory":
            raise IsADirectoryError(f"Cannot download a directory: {remote_path}")

        content = data.get("content", "")
        fmt = data.get("format")

        if fmt == "base64":
            content_bytes = base64.b64decode(content)
        else:
            # Assume text if it's not base64 explicitly encoded
            content_bytes = str(content).encode("utf-8")

        with open(local_path, "wb") as f:
            f.write(content_bytes)

    def rm(self, remote_path: str):
        self._request("DELETE", remote_path)
