# skill_loader.py
import os
import importlib.util
import inspect
from pathlib import Path
from smolagents import Tool

def load_skills_from_directory(skills_root_dir: str, model=None) -> list[Tool]:
    """
    遍历指定目录，动态加载所有符合结构的 Skill 并实例化。
    
    :param skills_root_dir: 存放所有 skills 的根目录 (例如 "./my_skills")
    :param model: 传递给 Tool 的 LLM 模型实例 (依赖注入)
    :return: 实例化后的 Tool 列表
    """
    loaded_tools = []
    root_path = Path(skills_root_dir)

    if not root_path.exists():
        print(f"⚠️ Warning: Skills directory '{skills_root_dir}' does not exist.")
        return []

    # 1. 遍历根目录下的所有子文件夹 (每个子文件夹作为一个 Skill)
    for skill_dir in root_path.iterdir():
        if skill_dir.is_dir() and not skill_dir.name.startswith(('.', '__')):
            
            # 2. 定位 scripts 目录下的 python 文件
            scripts_dir = skill_dir / "scripts"
            if not scripts_dir.exists():
                continue

            # 扫描 scripts 里的所有 .py 文件
            for py_file in scripts_dir.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue

                try:
                    # 3. 动态导入模块
                    spec = importlib.util.spec_from_file_location(
                        f"skills.{skill_dir.name}.{py_file.stem}", 
                        py_file
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # 4. 查找模块中定义的 Tool 子类
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) 
                                and issubclass(obj, Tool) 
                                and obj is not Tool 
                                and obj.__module__ == module.__name__): # 确保类是在当前文件定义的
                                
                                print(f"   >> Discovered Tool: [{name}] in {skill_dir.name}")
                                
                                # 5. 实例化 Tool (自动注入 path 和 model)
                                # 假设你的 Tool __init__ 接受 model 和 skill_path_root
                                try:
                                    # 尝试注入 skill_path_root
                                    tool_instance = obj(model=model, skill_path_root=str(skill_dir))
                                except TypeError:
                                    # 如果这个 Tool 不需要 skill_path_root (比如普通 Tool)，则回退
                                    tool_instance = obj(model=model)
                                    
                                loaded_tools.append(tool_instance)
                                
                except Exception as e:
                    print(f"❌ Error loading skill '{skill_dir.name}': {e}")

    return loaded_tools