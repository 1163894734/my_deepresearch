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
    """
    将普通函数适配为 Tool 对象。
    
    :param func: 普通Python函数
    :param metadata: Skill的元数据
    :return: Tool 实例
    """
    sig = inspect.signature(func)
    inputs = {}
    
    # 从函数签名提取参数信息
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        # 从参数注解获取类型，默认为 string
        param_type = "string"
        if param.annotation != inspect.Parameter.empty:
            if isinstance(param.annotation, type):
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
        
        inputs[param_name] = {
            "type": param_type,
            "description": f"Parameter: {param_name}"
        }
    
    # 如果没有参数，添加默认的 input 参数
    if not inputs:
        inputs["input"] = {
            "type": "string",
            "description": "Input parameter"
        }
    
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
    """
    将普通类适配为 Tool 对象。
    假设类有 run() 或 execute() 或 __call__() 方法。
    
    :param cls: 普通Python类
    :param metadata: Skill的元数据
    :return: Tool 实例
    """
    instance = cls()
    
    # 查找可执行的方法
    execute_method = None
    for method_name in ["run", "execute", "__call__"]:
        if hasattr(instance, method_name):
            execute_method = getattr(instance, method_name)
            break
    
    if execute_method is None:
        raise ValueError(f"Class {cls.__name__} must have 'run', 'execute' or '__call__' method")
    
    sig = inspect.signature(execute_method)
    inputs = {}
    
    # 从方法签名提取参数信息
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
        
        inputs[param_name] = {
            "type": param_type,
            "description": f"Parameter: {param_name}"
        }
    
    # 如果没有参数，添加默认的 input 参数
    if not inputs:
        inputs["input"] = {
            "type": "string",
            "description": "Input parameter"
        }
    
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

    支持三种 skill 类型：
    1. 继承 Tool 类的脚本 (最灵活)
    2. 包含普通函数或类的脚本 (自动适配)
    3. 仅包含 SKILL.md 的 prompt-only skill

    加载时只解析 SKILL.md 中的 name/description；
    prompt-only skill 在调用时读取完整 SKILL.md 作为提示词。

    :param skills_root_dir: 存放所有 skills 的根目录 (例如 "./my_skills")
    :param model: 传递给 Tool 的 LLM 模型实例 (依赖注入)
    :return: 实例化后的 Tool 列表
    """
    loaded_tools: list[Tool] = []
    root_path = Path(skills_root_dir)

    if not root_path.exists():
        print(f"⚠️ Warning: Skills directory '{skills_root_dir}' does not exist.")
        return []

    # 1. 遍历根目录下的所有子文件夹 (每个子文件夹作为一个 Skill)
    for skill_dir in root_path.iterdir():
        if not skill_dir.is_dir() or skill_dir.name.startswith((".", "__")):
            continue

        metadata = _extract_metadata(skill_dir)

        # 2. 定位 scripts 目录下的 python 文件
        scripts_dir = skill_dir / "scripts"
        if not scripts_dir.exists():
            # prompt-only skill
            if metadata.skill_md_path and metadata.skill_md_path.exists():
                loaded_tools.append(PromptSkillTool(metadata=metadata, model=model))
            continue

        # 扫描 scripts 里的所有 .py 文件
        for py_file in scripts_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                # 3. 动态导入模块
                spec = importlib.util.spec_from_file_location(
                    f"skills.{skill_dir.name}.{py_file.stem}",
                    py_file,
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    tool_found = False
                    
                    # 4.1 首先查找模块中定义的 Tool 子类 (优先级最高)
                    for _, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, Tool)
                            and obj is not Tool
                            and obj.__module__ == module.__name__
                        ):  # 确保类是在当前文件定义的

                            print(f"   >> Discovered Tool: [{obj.__name__}] in {skill_dir.name}")

                            # 5. 实例化 Tool (自动注入 path 和 model)
                            try:
                                tool_instance = obj(model=model, skill_path_root=str(skill_dir))
                            except TypeError:
                                # 如果这个 Tool 不需要 skill_path_root，则回退
                                tool_instance = obj(model=model)

                            loaded_tools.append(tool_instance)
                            tool_found = True
                    
                    # 4.2 如果没找到 Tool 子类，则尝试适配普通函数或类
                    if not tool_found:
                        for obj_name, obj in inspect.getmembers(module):
                            # 跳过内置对象和导入的对象
                            if obj_name.startswith("_") or obj.__module__ != module.__name__:
                                continue
                            
                            # 尝试适配普通函数
                            if inspect.isfunction(obj):
                                print(f"   >> Discovered function: [{obj_name}] in {skill_dir.name}, auto-adapting...")
                                try:
                                    adapted_tool = _adapt_function_to_tool(obj, metadata)
                                    loaded_tools.append(adapted_tool)
                                    tool_found = True
                                except Exception as e:
                                    print(f"      ⚠️ Failed to adapt function '{obj_name}': {e}")
                            
                            # 尝试适配普通类
                            elif inspect.isclass(obj):
                                print(f"   >> Discovered class: [{obj_name}] in {skill_dir.name}, auto-adapting...")
                                try:
                                    adapted_tool = _adapt_class_to_tool(obj, metadata)
                                    loaded_tools.append(adapted_tool)
                                    tool_found = True
                                except Exception as e:
                                    print(f"      ⚠️ Failed to adapt class '{obj_name}': {e}")
                    
                    if not tool_found:
                        print(f"   ⚠️ No Tool, function, or class found in {py_file}")

            except Exception as e:
                print(f"❌ Error loading skill '{skill_dir.name}': {e}")

    return loaded_tools