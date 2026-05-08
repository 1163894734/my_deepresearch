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
    skill_type: str = "prompt"
    enabled: bool = True

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
    skill_type = "sop" # 默认值
    enabled = True
    
    if skill_md_path.exists():
        content = skill_md_path.read_text(encoding="utf-8")
        metadata, body = _split_frontmatter(content)
        raw_name = metadata.get("name", "").strip()
        raw_description = metadata.get("description", "").strip()
        skill_type = metadata.get("type", "sop").strip().lower() # 读取 type 字段
        raw_enabled = metadata.get("enabled", "true").strip().lower() 
        enabled = raw_enabled != "false" 
        
        if not raw_description:
            for line in body.splitlines():
                stripped = line.strip()
                if stripped:
                    raw_description = stripped
                    break

    name = _sanitize_tool_name(raw_name, skill_dir.name)
    description = raw_description or f"Skill loaded from {skill_dir.name}."
    return SkillMetadata(name=name, description=description, skill_dir=skill_dir, skill_md_path=skill_md_path, skill_type=skill_type, enabled=enabled)


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


class SopSkillTool(Tool):
    """
    专门用于读取并返回 SKILL.md 正文的工具。
    """
    output_type = "string"
    inputs = {}  # 零输入参数

    def __init__(self, metadata: SkillMetadata):
        super().__init__()
        self.name = metadata.name
        self._skill_md_path = metadata.skill_md_path
        self._skill_dir = metadata.skill_dir  # 👈 记录技能目录，方便后续找 assets
        
        # 👇【新增核心逻辑 1】：自动探测 assets 目录并提取绝对路径
        assets_dir = self._skill_dir / "assets"
        assets_hint = ""
        if assets_dir.exists() and assets_dir.is_dir():
            abs_assets_path = assets_dir.resolve()
            assets_hint = f"\n💡【系统提示】：配套写作模版已就绪，位于绝对路径 `{abs_assets_path}`，请在需要时直接使用 open() 函数前往该目录读取。\n"

        self.description = (
            f"{metadata.description}\n"
            f"{assets_hint}"  # 👈 将绝对路径提示注入到工具的 description 中
            f"[⚠️系统警告⚠️]：调用此工具将返回一份 SOP(操作指南) 或指令文档。"
            f"你必须仔细阅读返回的文档，并使用其他代码/工具去一步步执行文档里的要求。"
            f"**绝对严禁**将返回的指南内容直接作为 final_answer 输出给用户！"
        )

    def forward(self) -> str:
        if not self._skill_md_path or not self._skill_md_path.exists():
            return "SOP 文件不存在。"

        raw_content = _read_skill_prompt(self._skill_md_path)
        
        # 🌟 动态扫描该技能的 assets 目录
        assets_text = ""
        assets_dir = self._skill_dir / "assets"
        if assets_dir.exists() and assets_dir.is_dir():
            # 过滤掉隐藏文件 (如 .DS_Store)
            assets_list = [f.name for f in assets_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
            if assets_list:
                # 👇【新增核心逻辑 2】：在 SOP 返回正文中也把绝对路径塞进去，双重保险
                abs_assets_path = assets_dir.resolve()
                assets_text = "\n\n📌 【当前技能专属资产 (Assets) 清单】\n"
                assets_text += f"本技能附带以下模板文件，所在绝对路径为：`{abs_assets_path}`\n"
                assets_text += "你可以直接使用 Python 的 `open()` 函数拼接绝对路径进行读取，或者使用 `read_skill_asset` 工具：\n"
                for asset in assets_list:
                    assets_text += f"  - 文件名: {asset} (如用工具，参数为: skill_name='{self._skill_dir.name}', asset_name='{asset}')\n"
        
        guarded_response = f"""
================================================================================
【AGENT 内部执行手册 - 绝密 - 严禁输出给用户】
以下是供你（Agent）阅读和执行的操作指南（SOP）。
请阅读后，立即思考第一步该写什么代码或调用什么工具，去实际完成用户的任务。
严禁使用 final_answer 直接复述本手册的内容！{assets_text}
================================================================================

{raw_content}

================================================================================
【手册结束】现在，请根据上述流程，开始你的思考(Thought)并执行实际动作(Action)！
================================================================================
"""
        return guarded_response.strip()


# 🌟 新增工具：全局资产读取器
class ReadSkillAssetTool(Tool):
    """
    精确读取特定技能的 assets 目录下的文件内容。
    """
    name = "read_skill_asset"
    description = "当你在 SOP 指南中看到需要参考特定的模板、规则或数据文件时，使用此工具读取其完整内容。"
    
    inputs = {
        "skill_name": {
            "type": "string",
            "description": "当前正在执行的技能名称（即技能所在的文件夹名称）"
        },
        "asset_name": {
            "type": "string",
            "description": "要读取的模板文件名，例如 'format_rules.json'"
        }
    }
    
    output_type = "string"

    def __init__(self, skills_root_dir: str):
        super().__init__()
        self.skills_root_dir = Path(skills_root_dir).resolve()

    def forward(self, skill_name: str, asset_name: str) -> str:
        target_path = (self.skills_root_dir / skill_name / "assets" / asset_name).resolve()
        expected_assets_dir = (self.skills_root_dir / skill_name / "assets").resolve()
        
        # 安全防御：防路径穿越漏洞 (比如输入 asset_name="../../etc/passwd")
        try:
            target_path.relative_to(expected_assets_dir)
        except ValueError:
            return f"❌ 错误：非法的文件访问请求，禁止跨目录读取 ({asset_name})。"
            
        if not target_path.exists() or not target_path.is_file():
            return f"❌ 错误：在技能 '{skill_name}' 的 assets 目录下未找到文件 '{asset_name}'。"
            
        try:
            content = target_path.read_text(encoding='utf-8')
            return f"✅ 成功读取模板【{skill_name}/assets/{asset_name}】，内容如下：\n\n{content}"
        except Exception as e:
            return f"❌ 错误：读取资产文件失败，原因：{str(e)}"


def _adapt_function_to_tool(func: Any, metadata: SkillMetadata) -> Tool:
    # (原有代码保持不变) ...
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
    # (原有代码保持不变) ...
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
    """
    loaded_tools: list[Tool] = []
    root_path = Path(skills_root_dir)

    print(f"🔍 [Skill Loader] 开始扫描技能目录: {root_path.absolute()}")

    if not root_path.exists():
        print(f"⚠️ [Skill Loader] 警告: 目录 '{skills_root_dir}' 不存在。")
        return []

    # 🌟 新增逻辑：将全局模板读取工具预先塞入工具列表
    global_asset_tool = ReadSkillAssetTool(skills_root_dir=skills_root_dir)
    loaded_tools.append(global_asset_tool)
    print(f"  ┣━ 🗂️  [Global Tool] {global_asset_tool.name:<28} (Global Asset Reader)")

    for skill_dir in root_path.iterdir():
        if not skill_dir.is_dir() or skill_dir.name.startswith((".", "__")):
            continue

        metadata = _extract_metadata(skill_dir)
        if not metadata.enabled:
            print(f"  ┣━ ⏭️  [Disabled]    {metadata.name:<28} (from {skill_dir.name})")
            continue
        scripts_dir = skill_dir / "scripts"

        # 1. 只要存在 SKILL.md，就先注册 SOP 或 Prompt 工具
        if metadata.skill_md_path and metadata.skill_md_path.exists():
            if metadata.skill_type == "sop":
                loaded_tools.append(SopSkillTool(metadata=metadata))
                print(f"  ┣━ 📖 [SOP Tool]    {metadata.name:<28} (from {skill_dir.name})")
            else:
                loaded_tools.append(PromptSkillTool(metadata=metadata, model=model))
                print(f"  ┣━ 📝 [Prompt Tool] {metadata.name:<28} (from {skill_dir.name})")

        # 2. 如果存在 scripts 目录，继续将 Python 代码注册为全局工具
        if not scripts_dir.exists():
            continue

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
                            if obj_name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
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