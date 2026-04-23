import os
import time
import logging
from smolagents import CodeAgent
from scripts.skill_loader import load_skills_from_directory
from scripts.custom_tools import SaveFileTool

# 导入前沿技术分析专属工具
from scripts.frontier_tools import (
    FragmentedInfoAggregatorTool,
    KeyFrontierTechnologyMiningTool,
    WeakSignalTechnologyMiningTool,
    EvolutionPathAnalysisTool,
    KeyTechnologyEvolutionSensingTool
)

# 假设刚才生成的多源采集工具保存在了这里，如果没有请将其补充进 frontier_tools.py
from scripts.frontier_tools import MultiSourceDataCollectorTool 

import utils.common_utils as common_utils

# =========================
# 1. 基础环境与日志初始化
# =========================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
run_timestamp = time.strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(base_dir, "outputs", f"frontier_sensing_{run_timestamp}")

os.makedirs(run_dir, exist_ok=True)
os.makedirs(os.path.join(run_dir, "reports"), exist_ok=True)

smol_logger = logging.getLogger("smolagents")
smol_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join(run_dir, "agent_run.log"), encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
smol_logger.addHandler(file_handler)

def main():
    print(f"📁 前沿技术态势感知专属目录已创建: {run_dir}")
    model = common_utils.ModelProvider.get_model()

    # =========================
    # 2. 挂载底层工具与技能 (SOP)
    # =========================
    tools = [
        MultiSourceDataCollectorTool(),
        FragmentedInfoAggregatorTool(),
        KeyFrontierTechnologyMiningTool(),
        WeakSignalTechnologyMiningTool(),
        EvolutionPathAnalysisTool(),
        KeyTechnologyEvolutionSensingTool(),
        SaveFileTool()
    ]
    
    # 加载包含 frontier_tech_sensing_sop 的所有 SOP 指南
    skills = load_skills_from_directory("skills", model=model)
    tools.extend(skills)

    # =========================
    # 3. 实例化单脑总控 (Single-Brain Director)
    # =========================
    director_agent = CodeAgent(
        name="tech_sensing_director",
        description="负责前沿技术态势感知的单脑总管。必须严格按照 `frontier_tech_sensing_sop` 中的步骤调用。",
        tools=tools,
        model=model,
        instructions="""你是一个高级 AI 技术情报总管。
        每次执行任务前，务必先调用 `frontier_tech_sensing_sop` 工具，严格按照里面的 5 个阶段步骤调度工具。
        【架构注意】：系统采用单脑流水线架构，你需要利用 Python 沙盒将上一步工具返回的数据结构（如 table）精准传递给下游工具。
        【格式注意】：所有最终报告必须使用全中文（简体）。""",
        additional_authorized_imports=["json", "time", "os", "pandas"]
    )

    # 注入全局上下文变量
    target_topic = "脑机接口"
    director_agent.state["target_topic"] = target_topic
    director_agent.state["run_dir"] = run_dir

    try:
        smol_logger.info("🎬 Director Agent 启动全链路前沿技术态势感知引擎...")
        
        # 触发执行
        director_agent.run("请根据注入的 target_topic 变量，启动前沿技术态势感知 SOP 分析流程。")
        
        print(f"\n🎉 研判完成！完整报告及过程数据已落盘至: {run_dir}")
    except Exception as e:
        smol_logger.error(f"❌ 发生致命错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()