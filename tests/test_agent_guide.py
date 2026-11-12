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

"""`colab agent-guide` must work as a guide AND as skill data."""

import json
import os

import pytest
from typer.testing import CliRunner

from colab_cli.cli import app
from colab_cli.quickstart import (
    QUICKSTART,
    skill_document,
    split_frontmatter,
)

runner = CliRunner()


# ----------------------------------------------------------- frontmatter


def test_split_frontmatter_reads_name_and_description():
    content = (
        "---\n"
        "name: colab-operator\n"
        "description: Operate Google Colab runtimes with the CLI.\n"
        "---\n"
        "\n# Body\n"
    )
    meta, body = split_frontmatter(content)
    assert meta["name"] == "colab-operator"
    assert meta["description"] == "Operate Google Colab runtimes with the CLI."
    assert body.strip().startswith("# Body")


def test_split_frontmatter_without_frontmatter_returns_the_content():
    content = "# Just markdown\n"
    meta, body = split_frontmatter(content)
    assert meta == {}
    assert body == content


def test_skill_document_has_the_fields_a_harness_needs():
    doc = skill_document(
        "---\nname: colab-operator\ndescription: Does things.\n---\n\n# Body\n",
        quickstart=QUICKSTART,
    )
    assert doc["name"] == "colab-operator"
    assert doc["description"] == "Does things."
    assert doc["content"].strip().startswith("# Body")
    assert doc["quickstart"] == QUICKSTART


# ------------------------------------------------------------ the command


def test_agent_guide_quickstart_is_a_recipe():
    result = runner.invoke(app, ["agent-guide", "--quickstart"])
    assert result.exit_code == 0, result.output
    text = result.stdout
    # The recipe must name the actual commands, in order.
    for fragment in [
        "colab doctor",
        "colab new -s proj",
        "colab exec -s proj",
        "colab status -s proj",
        "colab stop -s proj",
    ]:
        assert fragment in text, f"missing {fragment!r} from the quick-start"


def test_agent_guide_quickstart_has_a_symptom_table():
    """A model needs to map an error to an action."""
    result = runner.invoke(app, ["agent-guide", "--quickstart"])
    assert "No active sessions found" in result.stdout
    assert "Allocation refused" in result.stdout


def test_agent_guide_json_is_skill_data():
    result = runner.invoke(app, ["agent-guide", "--json"])
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["name"]
    assert doc["description"]
    assert len(doc["content"]) > 500
    assert doc["quickstart"]


def test_agent_guide_install_writes_a_skill_file(tmp_path):
    result = runner.invoke(app, ["agent-guide", "--install", str(tmp_path)])
    assert result.exit_code == 0, result.output

    written = list(tmp_path.rglob("SKILL.md"))
    assert len(written) == 1
    target = written[0]
    # The directory is named after the skill.
    assert target.parent.name == "colab-operator"

    content = target.read_text(encoding="utf-8")
    # The frontmatter must come first.
    assert content.startswith("---\n")
    assert "name: colab-operator" in content
    # The quick-start comes before the long handbook.
    assert content.index("colab doctor") < content.index("# Full handbook")


def test_agent_guide_install_can_rename_the_skill(tmp_path):
    result = runner.invoke(
        app,
        ["agent-guide", "--install", str(tmp_path), "--skill-name", "my-colab"],
    )
    assert result.exit_code == 0, result.output
    written = list(tmp_path.rglob("SKILL.md"))
    assert written[0].parent.name == "my-colab"


def test_help_points_at_the_guide():
    """The model has to be able to find the guide from the help."""
    result = runner.invoke(app, ["help"])
    assert result.exit_code == 0
    assert "agent-guide" in result.stderr + result.stdout
    assert "colab doctor" in result.stderr + result.stdout
