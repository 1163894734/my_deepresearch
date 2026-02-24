from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "open_deep_research"
SCRIPTS_DIR = EXAMPLES_DIR / "scripts"
sys.path.append(str(SCRIPTS_DIR))

from skill_loader import PromptSkillTool, _extract_metadata, _read_skill_prompt  # noqa: E402


@dataclass
class _FakeResponse:
    content: str


class _FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return _FakeResponse(content="OK")


def test_extract_metadata_from_skill_md():
    skill_dir = EXAMPLES_DIR / "skills" / "code_explainer"
    metadata = _extract_metadata(skill_dir)

    assert metadata.name == "code_explainer"
    assert metadata.description
    assert metadata.skill_md_path is not None


def test_read_skill_prompt_body():
    skill_md_path = EXAMPLES_DIR / "skills" / "code_explainer" / "SKILL.md"
    prompt_body = _read_skill_prompt(skill_md_path)

    assert "解释代码时" in prompt_body


def test_prompt_skill_tool_uses_skill_prompt():
    skill_dir = EXAMPLES_DIR / "skills" / "code_explainer"
    metadata = _extract_metadata(skill_dir)
    fake_model = _FakeModel()
    tool = PromptSkillTool(metadata=metadata, model=fake_model)

    output = tool.forward("请解释这段代码")

    assert output == "OK"
    assert fake_model.calls
    messages, _ = fake_model.calls[0]
    system_message = messages[0]
    user_message = messages[1]

    assert "解释代码时" in system_message.content[0]["text"]
    assert "请解释这段代码" in user_message.content[0]["text"]
