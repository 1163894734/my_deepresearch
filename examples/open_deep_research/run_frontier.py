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
    # 调试用的任务列表，正式运行时使用下面的查找sop版本
    search_tasks = """
请按以下步骤顺序执行任务，每个步骤的输出需通过print打印，**严禁省略任何步骤**：
【注意】`run_dir` 变量已注入环境，指向当前运行目录，请直接使用，不要通过工具获取该变量的值，并且请不要写成"run_dir"这样的字符串。
0. 定义变量(必须执行)：
origin_paper_dir = "../paper_db"
paper_meta_dir = run_dir + "/papers"
3. 技术主题聚类分析：
从`run_dir`目录下读取1_heterogeneous_data.json文件（内容为列表，每个元素是论文元信息字典）。打印第一条数据并简要分析其结构。从每条数据中提取与内容高度相关的字段（至少包含：'标题'、'关键词'、'所属技术领域'）。将这些提取的字段作为参数调用technology_clusterer工具进行聚类。聚类结果中每个类别包含"cluster_label"字段。将聚类结果保存到`run_dir`目录下，文件名为2_heterogeneous_clusters.json。打印输出聚类簇的个数及每个簇的基本信息。

4. 技术信号评估：
从`run_dir`目录下读取2_heterogeneous_clusters.json文件（内容为列表，每个元素是一个聚类簇，簇包含papers字段记录该簇的论文列表）。调用tech_signal_evaluator工具对这些聚类簇进行技术信号评估（如成熟度、影响力、活跃度等）。将工具输出保存为3_technology_signals.json文件到`run_dir`目录。打印评估完成信息和关键指标摘要。

5. 技术内涵提取：
从`run_dir`目录下读取3_technology_signals.json文件。调用tech_connotation_extractor工具提取每个技术簇的技术内涵（如核心技术原理、关键使能技术、应用场景等）。将结果保存为4_technology_connotations.json文件到`run_dir`目录。打印提取完成信息。

6. 技术演进路径分析：
从`run_dir`目录下读取4_technology_connotations.json文件。调用tech_evolution_analyzer工具分析技术演进路径（如技术代际划分、关键里程碑、发展趋势等）。将结果保存为5_technology_evolution.json文件到`run_dir`目录。打印分析完成信息及演进路径概要。

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

    search_tasks = "请查找对应sop 执行前沿技术进展分析"

    # 设置运行目录并执行
    director_agent.state["run_dir"] = run_dir
    director_agent.run(f"请执行以下任务：\n{search_tasks}")

if __name__ == "__main__":
    main()