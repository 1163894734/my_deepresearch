import os
import logging
from scripts.skill_loader import load_skills_from_directory
from utils.agent_helper import setup_run_env
from utils import common_utils
from smolagents import CodeAgent
# 引入我们刚写的新工具
from scripts.frontier_tools import *
from scripts.custom_tools import LocalPaperInjectorTool, SaveFileTool,LoadFileTool, SetVariableTool, GetVariableTool, MdToWordTool

run_dir, logger = setup_run_env("frontier_identify",run_timestamp="20260519_151853")
# run_dir, logger = setup_run_env("frontier_identify",run_timestamp="20260517_150338")

def main():
    model = common_utils.ModelProvider.get_model()
    skills = load_skills_from_directory("skills", model=model)
    tools=[AcademicSearchTool(), SaveFileTool(), LLMDataMappingTool(),LoadFileTool(), TechnologyClustererTool(), TechSignalEvaluatorTool(), TechConnotationExtractorTool(), TechEvolutionAnalyzerTool(),LocalPaperInjectorTool(), SetVariableTool(), GetVariableTool(),FrontierReportGeneratorTool(), WeakSignalReportGeneratorTool(), MdToWordTool()]
    tools.extend(skills)
    # 极简 SOP：只需调用一次工具
    SOP_INSTRUCTIONS = """
    """
    
    director_agent = CodeAgent(
        name="director_agent",
        description="最高总管。负责接收检索任务并调用一键下载工具。",
        tools=tools,
        model=model,
        instructions=SOP_INSTRUCTIONS,
        additional_authorized_imports=['json','re','os','posixpath']
    )
    
    search_tasks = """
请按以下步骤顺序执行任务，每个步骤的输出需通过print打印，**严禁省略任何步骤**：
【注意】`run_dir` 变量已注入环境，指向当前运行目录，请直接使用，不要通过工具获取该变量的值，并且请不要写成"run_dir"这样的字符串。
0. 定义变量(必须执行)：
origin_paper_dir = "../paper_db"
paper_meta_dir = run_dir + "/papers"
7.总结前沿技术：
从`run_dir`目录下读取5_technology_evolution.json文件。调用frontier_report_generator工具撰写《关键前沿技术识别与分析报告》
使用save_file工具将《关键前沿技术识别与分析报告》保存为`run_dir`目录下的`key_frontier_technologies_report.md`
调用weak_signal_report_generator工具撰写《弱信号技术识别与分析报告》
使用save_file工具将《弱信号技术识别与分析报告》保存为`run_dir`目录下的`weak_signal_technologies_report.md`

8.转为word格式输出：
调用markdown_to_word工具将`key_frontier_technologies_report.md`转换为`key_frontier_technologies_report.docx`，并保存到`run_dir`目录下
调用markdown_to_word工具将`weak_signal_technologies_report.md`转换为`weak_signal_technologies_report.docx`，并保存到`run_dir`目录下
并使用final_answer返回简短的完成信号结束整个流程。
"""

    # search_tasks = "请查找对应sop 执行前沿技术进展分析"

    # 设置运行目录并执行
    director_agent.state["run_dir"] = run_dir
    director_agent.run(f"请执行以下任务：\n{search_tasks}")

if __name__ == "__main__":
    main()