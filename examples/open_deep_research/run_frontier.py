import os
import logging
from utils import common_utils
from smolagents import CodeAgent
# 引入我们刚写的新工具
from scripts.frontier_tools import *
from scripts.custom_tools import SaveFileTool,LoadFileTool

base_dir = os.path.dirname(os.path.abspath(__file__))
time_str = "20260424_150000"
run_dir = f"/Users/wangchao/project/my_deepresearch/examples/frontier/run_{time_str}"

def main():
    model = common_utils.ModelProvider.get_model()
    
    # 极简 SOP：只需调用一次工具
    SOP_INSTRUCTIONS = """
    
    """
    
    director_agent = CodeAgent(
        name="director_agent",
        description="最高总管。负责接收检索任务并调用一键下载工具。",
        tools=[AcademicSearchTool(), SaveFileTool(), LLMDataMappingTool(),LoadFileTool(), TechnologyClustererTool(), TechSignalEvaluatorTool(), TechConnotationExtractorTool(), TechEvolutionAnalyzerTool()], # 只需要这一个工具！
        model=model,
        instructions=SOP_INSTRUCTIONS,
        additional_authorized_imports=['json','re','os','posixpath']
    )
    
    search_tasks = """
    请按以下步骤顺序执行任务，每个步骤的输出需通过print打印：
    【注意】`run_dir` 变量已注入环境，指向当前运行目录，请直接使用，不要通过工具获取该变量的值。
    1. 解析论文检索任务并批量检索保存：
    解析以下多行文本，每行包含技术主题和对应的检索式。对每一行，调用academic_search(source="openalex", query=检索式, per_page=10)进行检索，然后将检索结果通过save_file工具保存到`run_dir`的`papers`目录下（文件名自定，如按主题命名）。完成后print输出已保存的文件列表。
    
    检索任务列表：
    硅通孔	TS=("3DHI" OR "3D Heterogeneous Integration") AND TS=("Through-Silicon Via" OR "TSV")
    混合键合	TS=("3DHI" OR "3D Heterogeneous Integration") AND TS=("Hybrid Bonding")
    微凸点	TS=("3DHI" OR "3D Heterogeneous Integration") AND TS=("Micro-bump")
    晶圆级封装	TS=("3DHI" OR "3D Heterogeneous Integration") AND TS=("Wafer-Level Packaging" OR "WLP")
    晶圆对晶圆键合	TS=("3DHI" OR "3D Heterogeneous Integration") AND TS=("Wafer-to-Wafer" OR "W2W")
    芯片对晶圆键合	TS=("3DHI" OR "3D Heterogeneous Integration") AND TS=("Die-to-Wafer" OR "D2W")
    重布线层	TS=("3DHI" OR "3D Heterogeneous Integration") AND TS=("Redistribution Layer" OR "RDL")
    芯粒	TS=("3DHI" OR "3D Heterogeneous Integration") AND TS=("Chiplet")
    信号完整性	TS=("3DHI" OR "3D Heterogeneous Integration") AND TS=("Signal Integrity" OR "SI")
    热管理	TS=("3DHI" OR "3D Heterogeneous Integration") AND TS=("Thermal Management" OR "Heat Dissipation")
    电子设计自动化	TS=("3DHI" OR "3D Heterogeneous Integration") AND TS=("Electronic Design Automation" OR "EDA")

    2. 合并所有检索结果并标准化字段：
    遍历`run_dir`目录下的papers目录中的所有文件，用load_file读取每个文件（结果为字典，包含results字段，该字段为论文元信息列表）。将所有文件的results列表拼接成一个总列表。然后调用heterogeneous_data_mapping工具对总列表进行字段转化（统一字段命名），最后将转化后的结果使用save_file工具保存到`run_dir`目录下，文件名为heterogeneous_data.json。打印保存成功信息及记录总数。

    3. 技术主题聚类分析：
    从`run_dir`目录下读取heterogeneous_data.json文件（内容为列表，每个元素是论文元信息字典）。打印第一条数据并简要分析其结构。从每条数据中提取与内容高度相关的字段（至少包含：'标题'、'摘要'、'核心方法/方案总结'、'创新点'）。将这些提取的字段作为参数调用technology_clusterer工具进行聚类。聚类结果中每个类别包含"cluster_label"字段。将聚类结果保存到`run_dir`目录下，文件名为heterogeneous_clusters.json。打印输出聚类簇的个数及每个簇的基本信息。

    4. 技术信号评估：
    从`run_dir`目录下读取heterogeneous_clusters.json文件（内容为列表，每个元素是一个聚类簇，簇包含papers字段记录该簇的论文列表）。调用tech_signal_evaluator工具对这些聚类簇进行技术信号评估（如成熟度、影响力、活跃度等）。将工具输出保存为technology_signals.json文件到`run_dir`目录。打印评估完成信息和关键指标摘要。

    5. 技术内涵提取：
    从`run_dir`目录下读取technology_signals.json文件。调用tech_connotation_extractor工具提取每个技术簇的技术内涵（如核心技术原理、关键使能技术、应用场景等）。将结果保存为technology_connotations.json文件到`run_dir`目录。打印提取完成信息。

    6. 技术演进路径分析：
    从`run_dir`目录下读取technology_connotations.json文件。调用tech_evolution_analyzer工具分析技术演进路径（如技术代际划分、关键里程碑、发展趋势等）。将结果保存为technology_evolution.json文件到`run_dir`目录。打印分析完成信息及演进路径概要。
    """

    # 设置运行目录并执行
    director_agent.state["run_dir"] = run_dir
    director_agent.run(f"请执行以下任务：\n{search_tasks}")

if __name__ == "__main__":
    main()