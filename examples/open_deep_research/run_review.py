import os
import json
import time
import logging
from smolagents import CodeAgent, CustomAgent
from scripts.skill_loader import load_skills_from_directory
from scripts.custom_tools import *
# 引入所有所需工具 (合并自 run_write_agent.py)
from scripts.custom_tools import AcademicRAGTool, GenerateAbstractTool, GenerateBibliographyTool, GenerateConclusionTool, GetVariableTool, SaveFileTool, ParseJsonTool, FlattenOutlineTool, ReportAssemblerTool, SetVariableTool, UniversalRAGTool, LoadFileTool, MdToWordTool
import utils.common_utils as common_utils

# =========================
# 1. 基础配置与日志
# =========================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
run_timestamp = time.strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(base_dir, "outputs", f"deep_research_review_{run_timestamp}")
os.makedirs(run_dir, exist_ok=True)
os.makedirs(os.path.join(run_dir, "pdfs"), exist_ok=True)

smol_logger = logging.getLogger("smolagents")
smol_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
smol_logger.addHandler(console_handler)
file_handler = logging.FileHandler(os.path.join(run_dir, "agent_run.log"), encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
smol_logger.addHandler(file_handler)

def main():
    print(f"📁 通用深度研究专属目录已创建: {run_dir}")
    model = common_utils.ModelProvider.get_model()

    # =========================
    # 2. 实例化深度研究子智能体 (从 run_deepresearch.py 原封不动搬运)
    # =========================
    
    calibrator_agent = CustomAgent(
        model=model,
        tools=[AcademicSearchTool(),SetVariableTool(),GetVariableTool()], 
        name="calibrator_agent",
        description="用于在陌生领域快速建立认知锚点。必须传入 'task' 参数（宏观主题）。",
        instructions="""你是一个跨学科领域标定专家。面对一个陌生的领域，你需要快速建立认知体系。
        请严格按以下步骤执行：
        1. 调用 `tool_academic_search` 工具，使用检索词 "{task} comprehensive review OR state of the art survey" 获取最多 5 篇顶级文献，使用openalex进行检索。
        2. 阅读摘要后，提取出该领域的三个核心元数据。
        3. 【强制红线】使用 `tool_set_var` 工具将提取出的元数据存入变量 `calibrator_agent_result`，以供总控和其他智能体调用。存入的必须是一个纯净的 Python 字典对象（dict），格式如下：
        {
            "core_metrics": ["核心评价指标1", "指标2"],
            "dominant_paradigms": ["当前主导流派1", "流派2"],
            "critical_bottleneck": "最大的物理/工程/商业限制"
            "domain_limiters": ["aerospace", "aviation"]  # <=== 新增：提取2-3个该领域专属的排他性英文限定词，用于防止跨领域检索污染
        }
        4. 不要废话！通过 final_answer 提交的第三步的变量。
        """
    )

    forager_agent = CustomAgent(
        model=model,
        tools=[SetVariableTool(),GetVariableTool()],
        name="forager_agent",
        description="基于领域锚点，将宏观主题拆解为四维检索词。必须传入 'task' (主题) 和 'context' (标定字典)。",
        instructions="""你是一个高级情报检索专家。请结合用户的原主题和传入的 context，生成 8 个正交的英文检索 Query。
        必须严格覆盖以下四个通用维度（每个维度2个Query）：
        1. 历史范式转移 (Paradigm Shifts)
        2. context 中提及的 dominant_paradigms 的最新突破
        3. context 中提及的 critical_bottleneck 的失效分析与妥协方案
        4. 试图颠覆 context 中 core_metrics 的前沿黑天鹅技术
        【防漂移强制红线】：为了防止检索引擎返回其他领域的无关高引论文，你生成的每一个 Query 都【必须】包含领域限定后缀！
        生成检索词时核心专有名词必须使用双引号包裹以进行精确匹配
        具体做法：从 context 中提取 `domain_limiters`，并在每个 Query 末尾加上 `AND (limiter1 OR limiter2)`。
        例如：`historical paradigm shifts in composite materials AND (aerospace OR aviation)`。
        使用``tool_set_var``工具将生成的 8 个 Query 存入变量 `forager_agent_result`，格式必须是一个纯正的 Python List[str] 对象.
        5. 不要废话！通过 final_answer 提交的第四步的变量。
        """
    )

    analyst_agent = CustomAgent(
        model=model,
        tools=[ParseJsonTool(),SetVariableTool(),GetVariableTool()],
        name="analyst_agent",
        description="用于分析聚类后的情报数据。必须传入 'task' 参数。",
        instructions="""你是一个行业分析师。你将接收到文献聚类数据。请为每个聚类写一段 100 字的核心洞察。要求如下：
            1. 直接分析输入数据即可，绝不要调用任何外部工具或编写爬虫代码。
            2. 在输出完主题和瓶颈后，你必须作为裁判评估这些文献是否足够支撑写出一份深度大纲。
            3. 如果文献充分且覆盖了核心维度，请在最后单起一行输出：STATUS: PASS。
            4. 如果文献严重缺失某个关键维度（例如缺乏最新的破局点技术，或都是老旧文章），请在最后单起一行输出：STATUS: FAIL | MISSING: [用英文写出需要补充检索的1-2个具体关键词]。
            5. 使用 `tool_set_var` 工具将你的完整分析文本存入变量 `analyst_agent_result`，以供总控和其他智能体调用。最后通过 final_answer 提交任务完成的提示。""",
        additional_authorized_imports=["json", "collections"]
    )

    outliner_agent = CustomAgent(
        model=model,
        tools=[ParseJsonTool(),SetVariableTool(),GetVariableTool()],
        name="outliner_agent",
        description="基于洞察生成具有因果递进逻辑的三级大纲树。必须传入 'task' 参数。",
        instructions="""你是一个顶级综述架构师。请基于输入的洞察数据，构建深度调研的三级大纲。
        【逻辑红线】：大纲章节之间必须呈现强烈的因果递进关系（从旧范式 -> 当前主流 -> 痛点 -> 前沿破局点）。
        
        【纯逻辑架构与强制字段命名要求】：
        1. 你不需要在这一步分配具体的参考文献！你只需要专注于生成深度、详实的章节树。
        2. 嵌套子章节时，列表的键名必须严格命名为 `sections` 。
        3. 每一个最底层的章节节点，必须包含 `chapter_title` 和 `core_argument` (本节核心论点/摘要)。
        4. `core_argument` 请写得尽量详细，包含具体的学术关键词，因为后续的 RAG 引擎会使用这段话去全局文献库中进行精准召回。

        期望的完整JSON节点格式范例（必须严格遵循此结构）：
        [
            {
                "chapter_title": "1. 纳米材料技术概述",
                "sections": [
                    {
                        "chapter_title": "1.1 纳米尺度精准调控",
                        "core_argument": "探讨纳米材料如何实现靶向递送，分析其在物理层面的机制，以及当前面临的毒性挑战和失效分析。"
                    },
                    {
                        "chapter_title": "1.2 当前主流的制备工艺",
                        "core_argument": "详细论述CVD与原子层沉积（ALD）等主流工艺在良率与成本控制上的核心瓶颈与具体数据对比。"
                    }
                ]
            }
        ]
        
        请基于对材料的深度理解，尽可能把大纲写得细致。大纲必须是合法的JSON格式。
        使用 `tool_set_var` 工具将大纲对象存入变量 `outliner_agent_result`，最后使用 `final_answer` 宣告完成。""",
        additional_authorized_imports=["json", "collections"]
    )

    tools = [AcademicSearchTool(), InsightExtractorTool(), SemanticClusterTool(), PaperDownloaderTool(), SaveFileTool(), ParseJsonTool(),SetVariableTool(),GetVariableTool(),LocalPaperInjectorTool()]  # 基础工具
    skills = load_skills_from_directory("skills", model=model)
    tools.extend(skills)

    director_agent = CustomAgent(
        name="director_agent",
        description="负责深度研究的总体规划与调度。必须严格按照 `deep_research_director_sop` 中的步骤调用。",
        tools=tools,
        managed_agents=[calibrator_agent, forager_agent, analyst_agent, outliner_agent],
        model=model,
        instructions="""你是一个科研总架构师。配备了4个下级专家智能体。
        每次执行任务前，务必先调用 `deep_research_director_sop` 工具，严格按照里面的 6 个阶段步骤调度。
        【架构注意】：系统已升级为全局内存总线。巨型数组（如文献库）已自动存在内存中。请多使用 `tool_get_var` 获取数据，严禁在代码中 `print` 或硬编码巨长的文献 JSON 字符串！""",
        additional_authorized_imports=["json", "time", "ast", "re", "os"]
    )

    # =========================
    # 3. 实例化长文写作子智能体 (从 run_write_agent.py 原封不动搬运)
    # =========================
    
    section_writer_agent = CustomAgent(
        model=model,
        tools=[], 
        name="section_writer_agent",
        description="学术长文主笔。",
        instructions="""你是一名世界级的科技领域首席研究员。
        【任务】：根据传入的标题、核心论点和 RAG 语料，撰写学术报告的一个小节。
        【写作红线】：
        1. 严禁无营养废话。所有数据和结论必须直接来源于语料。
        2. 强制学术引用：必须在句末对所有使用的数据和观点进行文献引用。
           - 若语料来源为本地文献，使用 [paper_id]。
           - 若语料来源为在线网页，直接使用该 URL 作为引用标号（如 [https://www.mdpi.com/...]）。
        3. 绝对禁止输出“[注: 无本地文献支撑]”这类免责声明！只要 RAG 语料中提供了信息（不管是本地还是网页），请正常完成推演和引用。
        4. 严禁尝试调用 visit_webpage 等网页浏览工具！所有网页的内容系统已经提前抓取并放在了【RAG 精准提取语料】中，你只需要直接阅读 Prompt 里的文本。
        5. 格式要求：你必须且只能使用 python 代码块来提交结果，即 `final_answer("你的正文字符串")`，严禁直接输出纯文本。
        """
    )

    rag_tool = UniversalRAGTool()
    assembler_tool = ReportAssemblerTool(writer_agent=section_writer_agent, rag_tool=rag_tool)
    writer_tools = [ParseJsonTool(), FlattenOutlineTool(), assembler_tool, SaveFileTool(),SetVariableTool(),GetVariableTool(), GenerateBibliographyTool(), LoadFileTool(), GenerateAbstractTool(), GenerateConclusionTool(), MdToWordTool()]
    writer_tools.extend(skills)

    writer_director_agent = CustomAgent(
        name="writer_director_agent",
        description="最高总管。负责解析大纲并调用底层的自动组装引擎。",
        tools=writer_tools,
        model=model,
        additional_authorized_imports=["json"],
        instructions="""你是一个长文写作总架构师 (CodeAgent)。
        请从long_writer_section_sop获取写作指南，并严格按照指南要求执行代码，完成学术报告的撰写。
        """
    )

    # =========================
    # 4. 实例化顶层主控智能体 (Review Master)
    # =========================
    # 它本身没有任何业务提示词修改，只有调度代码
    review_agent = CodeAgent(
        name="review_agent",
        description="顶级综述主控引擎。端到端协调 Research 和 Write 两个阶段。",
        tools=[],
        managed_agents=[director_agent, writer_director_agent],
        model=model,
        instructions="""你是一个极简的总控调度智能体 (Review Agent)。你的唯一任务是按顺序触发两个子智能体，绝不要干涉它们的内部执行逻辑。
        请严格执行以下 Python 代码逻辑：
        1. 呼叫 `director_agent`，传入 task="请根据全局主题启动深度研究，并执行你的标准大纲生成SOP"，等待其完成。
        2. 呼叫 `writer_director_agent`，传入 task="大纲已就绪，请执行你的标准长文撰写、排版及导出SOP"，等待其完成。
        3. 运行结束后，使用 final_answer("全流程深度研究与综述撰写已成功完成！") 退出。
        
        【强制红线】：不要尝试去读取 json 文件，不要在代码里做状态传递 (state)，完全信任并放权给子智能体。
        """
    )

    # =========================
    # 5. 状态注入与任务启动
    # =========================
    target_topic = "三维异构集成"
    
    # 向所有需要环境信息的 Agent 注入一致的字典变量
    director_agent.state["target_topic"] = target_topic
    director_agent.state["run_dir"] = run_dir
    writer_director_agent.state["run_dir"] = run_dir
    review_agent.state["target_topic"] = target_topic
    review_agent.state["run_dir"] = run_dir

    try:
        smol_logger.info("🎬 Review Agent 端到端全流程主引擎启动...")
        review_agent.run("请依次调度 director_agent 和 writer_director_agent，完成从深度研究大纲生成到长文撰写的全流程。")
        print(f"\n🎉 规划与撰写全部完成！产出物已落盘至: {run_dir}")
    except Exception as e:
        smol_logger.error(f"❌ 发生致命错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()