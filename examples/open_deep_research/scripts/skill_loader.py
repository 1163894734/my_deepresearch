# skill_loader.py
import os
import importlib.util
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smolagents import Tool, ChatMessage, MessageRole
from smolagents.utils import is_valid_name

@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    skill_dir: Path
    skill_md_path: Path | None


def _split_frontmatter(markdown_text: str) -> tuple[dict[str, str], str]:
    lines = markdown_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, markdown_text

    end_index = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_index = idx
            break

    if end_index is None:
        return {}, markdown_text

    frontmatter_lines = lines[1:end_index]
    body = "\n".join(lines[end_index + 1 :]).lstrip()
    metadata: dict[str, str] = {}

    for line in frontmatter_lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"'")

    return metadata, body


def _sanitize_tool_name(raw_name: str, fallback_name: str) -> str:
    candidate = raw_name.strip() if raw_name else fallback_name
    candidate = re.sub(r"[^0-9a-zA-Z_]", "_", candidate)
    if candidate and candidate[0].isdigit():
        candidate = f"skill_{candidate}"
    if not is_valid_name(candidate):
        candidate = f"skill_{fallback_name}"
    return candidate


def _extract_metadata(skill_dir: Path) -> SkillMetadata:
    skill_md_path = skill_dir / "SKILL.md"
    raw_name = ""
    raw_description = ""
    if skill_md_path.exists():
        content = skill_md_path.read_text(encoding="utf-8")
        metadata, body = _split_frontmatter(content)
        raw_name = metadata.get("name", "").strip()
        raw_description = metadata.get("description", "").strip()
        if not raw_description:
            for line in body.splitlines():
                stripped = line.strip()
                if stripped:
                    raw_description = stripped
                    break

    name = _sanitize_tool_name(raw_name, skill_dir.name)
    description = raw_description or f"Skill loaded from {skill_dir.name}."
    return SkillMetadata(name=name, description=description, skill_dir=skill_dir, skill_md_path=skill_md_path)


def _read_skill_prompt(skill_md_path: Path) -> str:
    content = skill_md_path.read_text(encoding="utf-8")
    _, body = _split_frontmatter(content)
    return body.strip()


class PromptSkillTool(Tool):
    output_type = "string"
    inputs = {
        "input": {
            "type": "string",
            "description": "需要处理的用户输入或待解释的内容。",
        }
    }

    def __init__(self, metadata: SkillMetadata, model: Any):
        super().__init__()
        self.model = model
        self.name = metadata.name
        self.description = metadata.description
        self._skill_md_path = metadata.skill_md_path

    def forward(self, input: str) -> str:
        if not self.model:
            raise ValueError("PromptSkillTool requires a model instance.")
        if not self._skill_md_path or not self._skill_md_path.exists():
            raise FileNotFoundError("SKILL.md not found for prompt skill.")

        prompt_body = _read_skill_prompt(self._skill_md_path)
        system_prompt = prompt_body.strip()
        user_prompt = input.strip()

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": system_prompt}]),
            ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": user_prompt}]),
        ]
        response = self.model(messages)
        return response.content


def _adapt_function_to_tool(func: Any, metadata: SkillMetadata) -> Tool:
    sig = inspect.signature(func)
    inputs = {}
    
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        param_type = "string"
        if param.annotation != inspect.Parameter.empty:
            if isinstance(param.annotation, type):
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
        
        inputs[param_name] = {"type": param_type, "description": f"Parameter: {param_name}"}
    
    if not inputs:
        inputs["input"] = {"type": "string", "description": "Input parameter"}
    
    class AdaptedTool(Tool):
        output_type = "string"
    
    AdaptedTool.name = metadata.name
    AdaptedTool.description = metadata.description
    AdaptedTool.inputs = inputs
    
    def forward(self, **kwargs):
        return func(**kwargs)
    
    AdaptedTool.forward = forward
    return AdaptedTool()


def _adapt_class_to_tool(cls: Any, metadata: SkillMetadata) -> Tool:
    instance = cls()
    execute_method = None
    for method_name in ["run", "execute", "__call__"]:
        if hasattr(instance, method_name):
            execute_method = getattr(instance, method_name)
            break
    
    if execute_method is None:
        raise ValueError(f"Class {cls.__name__} must have 'run', 'execute' or '__call__' method")
    
    sig = inspect.signature(execute_method)
    inputs = {}
    
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        param_type = "string"
        if param.annotation != inspect.Parameter.empty:
            if isinstance(param.annotation, type):
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
        
        inputs[param_name] = {"type": param_type, "description": f"Parameter: {param_name}"}
    
    if not inputs:
        inputs["input"] = {"type": "string", "description": "Input parameter"}
    
    class AdaptedTool(Tool):
        output_type = "string"
    
    AdaptedTool.name = metadata.name
    AdaptedTool.description = metadata.description
    AdaptedTool.inputs = inputs
    
    def forward(self, **kwargs):
        return execute_method(**kwargs)
    
    AdaptedTool.forward = forward
    return AdaptedTool()


def load_skills_from_directory(skills_root_dir: str, model=None) -> list[Tool]:
    """
    遍历指定目录，动态加载所有符合结构的 Skill 并实例化。
    优化了日志展示，以树状结构清晰呈现不同类型的技能。
    """
    loaded_tools: list[Tool] = []
    root_path = Path(skills_root_dir)

    print(f"🔍 [Skill Loader] 开始扫描技能目录: {root_path.absolute()}")

    if not root_path.exists():
        print(f"⚠️ [Skill Loader] 警告: 目录 '{skills_root_dir}' 不存在。")
        return []

    for skill_dir in root_path.iterdir():
        if not skill_dir.is_dir() or skill_dir.name.startswith((".", "__")):
            continue

        metadata = _extract_metadata(skill_dir)
        scripts_dir = skill_dir / "scripts"

        # 1. 纯 Prompt 技能 (无 scripts 目录)
        if not scripts_dir.exists():
            if metadata.skill_md_path and metadata.skill_md_path.exists():
                loaded_tools.append(PromptSkillTool(metadata=metadata, model=model))
                print(f"  ┣━ 📝 [Prompt Tool] {metadata.name:<28} (from {skill_dir.name})")
            continue

        # 2. Python 代码技能
        for py_file in scripts_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                spec = importlib.util.spec_from_file_location(
                    f"skills.{skill_dir.name}.{py_file.stem}",
                    py_file,
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    tool_found = False
                    
                    # 4.1 查找 Tool 子类
                    for _, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, Tool)
                            and obj is not Tool
                            and obj.__module__ == module.__name__
                        ):
                            try:
                                tool_instance = obj(model=model, skill_path_root=str(skill_dir))
                            except TypeError:
                                tool_instance = obj(model=model)

                            loaded_tools.append(tool_instance)
                            tool_found = True
                            print(f"  ┣━ 🛠️  [Code Tool]   {obj.__name__:<28} (from {skill_dir.name})")
                    
                    # 4.2 适配普通函数或类
                    if not tool_found:
                        for obj_name, obj in inspect.getmembers(module):
                            if obj_name.startswith("_") or obj.__module__ != module.__name__:
                                continue
                            
                            if inspect.isfunction(obj):
                                try:
                                    adapted_tool = _adapt_function_to_tool(obj, metadata)
                                    loaded_tools.append(adapted_tool)
                                    tool_found = True
                                    print(f"  ┣━ ⚙️  [Func Tool]   {obj_name:<28} (from {skill_dir.name})")
                                except Exception as e:
                                    print(f"  ┣━ ❌ [Func Error]  {obj_name}: {e}")
                            
                            elif inspect.isclass(obj):
                                try:
                                    adapted_tool = _adapt_class_to_tool(obj, metadata)
                                    loaded_tools.append(adapted_tool)
                                    tool_found = True
                                    print(f"  ┣━ 📦 [Class Tool]  {obj_name:<28} (from {skill_dir.name})")
                                except Exception as e:
                                    print(f"  ┣━ ❌ [Class Error] {obj_name}: {e}")
                                    
            except Exception as e:
                print(f"  ┣━ ❌ [Load Error]  {skill_dir.name}: {e}")

    print(f"  ┗━ ✅ 共成功加载 {len(loaded_tools)} 个 Skill 工具\n")
    return loaded_tools