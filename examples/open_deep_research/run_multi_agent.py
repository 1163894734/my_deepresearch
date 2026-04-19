import os
import json
import time
import logging
from smolagents import CodeAgent, CustomAgent
from scripts.skill_loader import load_skills_from_directory
from scripts.custom_tools import *
import utils.common_utils as common_utils

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
run_timestamp = time.strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(base_dir, "outputs", f"deep_research_{run_timestamp}")
os.makedirs(run_dir, exist_ok=True)
os.makedirs(os.path.join(run_dir, "pdfs"), exist_ok=True)

smol_logger = logging.getLogger("smolagents")
smol_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join(run_dir, "agent_run.log"), encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
smol_logger.addHandler(file_handler)

def main():
    print(f"📁 通用深度研究专属目录已创建: {run_dir}")
    model = common_utils.ModelProvider.get_model()

    # =========================
    # 2. 实例化子智能体 (The Research Matrix)
    # =========================
    
    # 🌟 [新增] 0. 领域标定者 (Calibrator) 
    # 赋予它学术检索工具，让它自主去查顶级综述
    sub_agent = CustomAgent(
        model=model,
        tools=[], 
        name="sub_agent",
        description="用于在陌生领域快速建立认知锚点。必须传入 'task' 参数（宏观主题）。",
        # instructions="""使用代码`a=["123","张三","https://openalex.org/W3126951392"]`的格式，在 final_answer 中返回一个返回一个字符串: "前面的"+a+"后面的" """
        instructions="""使用代码`a={'key1':'123','key2':'张三'}`的格式，在 final_answer 中返回一个字符串: "前面的"+a+"后面的" """
    )
    director_agent = CodeAgent(
        name="director_agent",
        description="管理智能体",
        tools=[ParseJsonTool()],
        managed_agents=[sub_agent],
        model=model,
        instructions="""使用代码`tool_parse_json(b=sub_agent(task=task))`获取到如果是dict对象取key2的值，如果是list对象取第二个位置的值，并将该值作为最终答案通过 final_answer 输出""",
        additional_authorized_imports=["json", "time", "ast", "re", "os"]
    )

    # 测试通用性：这里可以换成任何陌生领域
    target_topic = "2025年航空材料领域的全貌（树脂基复合材料、CMC与高性能合金）"
    director_agent.state["target_topic"] = target_topic
    director_agent.state["run_dir"] = run_dir

    try:
        smol_logger.info("🎬 Director Agent 启动通用深度研究引擎...")
        director_agent.run("")
        print(f"\n🎉 规划完成！通用架构大纲已落盘至: {run_dir}/deep_research_outline.json")
    except Exception as e:
        smol_logger.error(f"❌ 发生致命错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()