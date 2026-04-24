import os
import json
import time
import logging
from smolagents import CodeAgent
from scripts.skill_loader import load_skills_from_directory
from smolagents.models import OpenAIModel
import utils.common_utils as common_utils
from multiple_tools import get_current_time, get_random_fact, search_wikipedia

# =========================
# 1. 初始化运行目录与全量日志
# =========================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
run_timestamp = time.strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(base_dir, "outputs", f"identification_{run_timestamp}")

def main():
    print(f"📁 本次任务专属目录已创建: {run_dir}")
    # =========================
    # 2. 加载技能与模型
    # =========================
    model = common_utils.ModelProvider.get_model()
    skills = load_skills_from_directory("skills", model=model)

    GENERIC_INSTRUCTIONS = """
    今天是什么天气
    """

    agent = CodeAgent(
        tools=skills + [],
        model=model,
        instructions=GENERIC_INSTRUCTIONS,
        additional_authorized_imports=["json", "time"]
    )

    agent.run()

if __name__ == "__main__":
    main()