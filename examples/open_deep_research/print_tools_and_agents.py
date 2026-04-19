import os
import time
import logging
from smolagents import CodeAgent
from scripts.skill_loader import load_skills_from_directory
import utils.common_utils as common_utils

# =========================
# 1. 初始化运行目录与全量日志
# =========================
base_dir = os.path.dirname(os.path.abspath(__file__))
run_timestamp = time.strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(base_dir, "outputs", f"identification_{run_timestamp}")
os.makedirs(run_dir, exist_ok=True)

# 捕获 smolagents 的原生日志
smol_logger = logging.getLogger("smolagents")
smol_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join(run_dir, "agent_run.log"), encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
smol_logger.addHandler(file_handler)

def main():
    print(f"📁 本次任务专属目录已创建: {run_dir}")
    smol_logger.info("🚀 启动前沿技术识别 Agentic Workflow...")

    # =========================
    # 2. 加载技能与模型
    # =========================
    model = common_utils.ModelProvider.get_model()
    skills = load_skills_from_directory("skills", model=model)

    GENERIC_INSTRUCTIONS = """
    你是一个高度自主的 AI 助手。你可以编写并在沙盒中运行 Python 代码来处理数据。
    当你遇到复杂的垂直业务请求时，请先检查工具列表中是否有对应的 SOP 或指南工具，并优先调用它们来了解你应该怎么做。
    """

    agent = CodeAgent(
        tools=skills,
        model=model,
        instructions=GENERIC_INSTRUCTIONS,
        additional_authorized_imports=["json", "time"]
    )

    def print_item(name: str, description: str) -> None:
        print(f"【{name}】")
        print(description)

    print("\n=== Tools ===")
    for tool in skills:
        print_item(tool.name, tool.description)

    print("\n=== Agents ===")
    print_item(agent.__class__.__name__, getattr(agent, "description", "No description available."))

if __name__ == "__main__":
    main()