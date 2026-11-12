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

"""The quick-start an agent needs, and the skill-document helpers.

``colab-tool agent-guide`` prints the full handbook (SKILL.md). This module holds
the short, imperative version -- the one to read when you have no context
and need to get a job running -- plus helpers for treating the handbook as
skill data (parse its frontmatter, emit it as JSON, install it into an
agent's skills directory).
"""

import re
from typing import Dict, Optional, Tuple

# The quick-start. Written as an ordered recipe so that a model with no
# context can follow it literally.
QUICKSTART = """\
# colab-tool quick-start (for an agent)

`colab-tool` rents a Google Colab CPU/GPU/TPU runtime and runs code on it.
Follow these steps in order. Each one says what the next is.

## 0. Check the setup (costs nothing)

    colab-tool doctor

It prints one verdict. If anything is broken it prints the exact commands to
fix it, numbered. Fix them and run it again. Do not start a job otherwise.

## 1. Rent a runtime

    colab-tool new -s proj --gpu T4

- Always pass `-s <name>`; an auto-generated name makes later commands ambiguous.
- `--gpu`: T4, L4, G4, H100, A100. Omit it for a CPU runtime.
- If the account's quota is spent, the CLI switches to another signed-in
  account and retries by itself. You do not need to handle that.
- If a matching runtime is already running, the CLI reuses it.

## 2. Run code

    colab-tool exec -s proj -f train.py --timeout 3600 --json

- `-f/--file` reads the script ON THIS MACHINE and sends its contents to the
  VM. It cannot run a file that only exists on the VM.
- `--timeout` is an IDLE limit: seconds allowed between two outputs. A script
  that prints every minute may run for hours. `0` means no limit.
- `--json` puts exactly one JSON envelope on stdout and the raw log on
  stderr, so stdout stays parseable.
- For a job that should clean up after itself: `colab-tool run --gpu T4 train.py`.

## 3. Collect what the job produced

    colab-tool status -s proj --json --files
    colab-tool download -s proj model.json ./model.json

`--files` lists /content with the paths that work, so you do not have to
guess where the checkpoint is.

## 4. Iterate

Edit the script locally, then run step 2 on the SAME session. The runtime,
the data on it and the installed packages all stay put.

## 5. Stop the runtime

    colab-tool stop -s proj

Always do this. A runtime left running keeps burning the account's daily
compute quota.

## When something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `No active sessions found` | The runtime was reclaimed (Colab frees idle VMs) or stopped | `colab-tool new -s proj --gpu T4` again |
| `File or directory not found` | The path is wrong, OR the runtime is gone | `colab-tool status -s proj` |
| `Allocation refused (precondition failed)` | This account's GPU quota/limit is spent | The CLI already tried other accounts; add one with `colab-tool account add <label>` |
| `No module named 'colab_cli'` | The code you sent imports the CLI's own package; it only exists locally | Remove that import |
| The run produced no output | It went `--timeout` seconds without printing | Raise `--timeout`, or make the script print progress |
| A large upload/download failed | The whole file goes in one base64 request | Prefer `colab-tool install -s proj <package>` (installs on the VM) |

## More

- `colab-tool agent-guide` -- the full handbook (this file is the short version).
- `colab-tool doctor --json` -- the same checks, machine-readable.
- `colab-tool help <command>` -- options for any command.
- `colab-tool account list` -- the signed-in accounts and which one is active.
"""

# Matches the YAML frontmatter of a skill document.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def split_frontmatter(content: str) -> Tuple[Dict[str, str], str]:
    """Split a skill document into its frontmatter mapping and its body.

    Returns ``({}, content)`` when the document has no frontmatter, so a
    caller never has to special-case it.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content

    meta: Dict[str, str] = {}
    key = None
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        if line[:1] in (" ", "\t"):
            # A continuation line of a multi-line scalar.
            if key:
                meta[key] = (meta.get(key, "") + " " + line.strip()).strip()
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        meta[key] = value
    return meta, content[match.end() :]


def skill_document(content: str, quickstart: str = "") -> Dict[str, str]:
    """Turn the handbook into skill data.

    The result is the shape an agent harness consumes: a name, a
    description (so the model can decide when to load it), the body, and
    optionally the quick-start as a separate, short field.
    """
    meta, body = split_frontmatter(content)
    doc = {
        "name": meta.get("name", "colab-operator"),
        "description": meta.get("description", ""),
        "content": body,
    }
    if quickstart:
        doc["quickstart"] = quickstart
    return doc
