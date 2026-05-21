import argparse
import os
import json
import time
from dotenv import load_dotenv

# 引入 smolagents 核心组件
from smolagents.agents import CustomAgent
from smolagents.default_tools import DuckDuckGoSearchTool
from utils import common_utils
from smolagents import CodeAgent
from smolagents.models import OpenAIModel

# 引入通用的技能加载器
from scripts.skill_loader import load_skills_from_directory

# ✨ 新增引入：从 agent_helper 引入环境初始化和记忆导出工具
from utils.agent_helper import setup_run_env, export_agent_memory
# ✨ 新增引入：从 custom_tools 引入学术搜索工具
from scripts.custom_tools import AcademicSearchTool

from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)

load_dotenv(override=True)

# ========== 浏览器相关的配置 ==========
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}
os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)
# ==============================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question", type=str, nargs="?", default="具身智能", help="需要解读的技术名词"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    target_term = args.question

    # =========================
    # 1. 初始化运行目录与全量日志 (✨ 使用 agent_helper 精简)
    # =========================
    # 这里的 prefix 使用 "interpretation"，以便和 deep_research 区分
    run_dir, smol_logger = setup_run_env("interpretation")
    smol_logger.info(f"🚀 启动技术名词解读 Agentic Workflow... 目标名词: {target_term}")

    # =========================
    # 2. 加载技能与模型
    # =========================
    model = common_utils.ModelProvider.get_model()
    
    # 动态加载所有的 skills，包括工具和 SOP
    skills = load_skills_from_directory("skills", model=model)

    GENERIC_INSTRUCTIONS = """
    你是一个高度自主的 AI 助手。你可以编写并在沙盒中运行 Python 代码来处理数据和调用各类工具。
    当你遇到复杂的垂直业务请求（如“技术名词解读”）时，请先检查工具列表中是否有对应的 SOP 或指南工具，并优先调用它们来了解你应该按什么标准流程执行。
    """

    # ==========================================
    # 1. 组建“基层调研团队” (专门对付网页)
    # ==========================================
    web_search_tool = DuckDuckGoSearchTool()
    web_search_tool.name = "web_search"
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    
    WEB_TOOLS = [web_search_tool, VisitTool(browser), PageDownTool(browser), FinderTool(browser)]

    search_agent = CustomAgent(
        model=model,
        tools=WEB_TOOLS,
        name="web_researcher",
        description="专门负责深度的网络搜索和长网页阅读。当你需要详细了解某个技术原理时，把任务交给他，他会自己搜索、阅读原文，并把总结好的事实返回给你。"
    )

    # ✨ 实例化学术检索工具
    academic_search_tool = AcademicSearchTool()

    # 实例化基于代码的 Agent (将学术搜索工具加入工具库)
    agent = CodeAgent(
        tools=skills + WEB_TOOLS + [academic_search_tool],
        managed_agents=[search_agent],
        model=model,
        instructions=GENERIC_INSTRUCTIONS,
        additional_authorized_imports=["json", "time", "os", "re", "concurrent.futures"]
    )

    # =========================
    # 3. 注入数据与环境变量
    # =========================
    # 将目标名词和目录注入到 state 中，方便 Agent 在编写 Python 代码时直接读取
    agent.state["target_term"] = target_term
    agent.state["run_dir"] = run_dir 

    # =========================
    # 4. 执行任务
    # =========================
    try:
        # 向 Agent 下发指令，Agent 会自主寻找 SOP 并执行
        prompt = f"请帮我执行【技术名词解读】任务。需要解读的目标概念是：{target_term}。请查阅相关SOP（如果存在），综合调用学术检索与网络搜索工具，最终生成百科版、专报版、科普版三版报告并保存到我指定的 run_dir 中。"
        
        result = agent.run(prompt)
        
        # ========== ✨ 记忆转存 (使用 agent_helper 统一精简) ==========
        export_agent_memory(agent, run_dir, "agent_thought_process.md")
        # =================================================================
        
        smol_logger.info("✅ 任务圆满完成。")
        print("\n================ 最终解读报告 ================\n")
        print(result)
        
    except Exception as e:
        smol_logger.error(f"❌ 执行过程中发生致命错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()