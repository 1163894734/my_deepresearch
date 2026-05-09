import os
import json
import time
import logging
from smolagents import CodeAgent, CustomAgent
from scripts.skill_loader import load_skills_from_directory
from scripts.custom_tools import *
import utils.common_utils as common_utils

# =========================
# 1. 基础配置与全局日志
# =========================
base_dir = os.path.dirname(os.path.abspath(__file__))
# 统一的时间戳作为本次端到端综述任务的专属文件夹
run_timestamp = time.strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(base_dir, "outputs", f"review_pipeline_{run_timestamp}")

os.makedirs(run_dir, exist_ok=True)
os.makedirs(os.path.join(run_dir, "pdfs"), exist_ok=True)

smol_logger = logging.getLogger("smolagents")
smol_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
smol_logger.addHandler(console_handler)
file_handler = logging.FileHandler(os.path.join(run_dir, "review_run.log"), encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
smol_logger.addHandler(file_handler)

def main():
    print(f"📁 端到端综述全流程专属目录已创建: {run_dir}")
    model = common_utils.ModelProvider.get_model()

    # ==========================================
    # 2. 实例化第一阶段智能体群 (Research Matrix)
    # ==========================================
    calibrator_agent = CustomAgent(
        model=model,
        tools=[AcademicSearchTool(), SetVariableTool(), GetVariableTool()], 
        name="calibrator_agent",
        description="用于在陌生领域快速建立认知锚点。必须传入 'task' 参数（宏观主题）。",
        instructions="""你是一个跨学科领域标定专家。面对一个陌生的领域，你需要快速建立认知体系。
        请严格按以下步骤执行：
        1. 调用 `tool_academic_search` 工具，使用检索词 "{task} comprehensive review OR state of the art survey" 获取最多 5 篇顶级文献。
        2. 阅读摘要后，提取出该领域的三个核心元数据。
        3. 【强制红线】使用 `tool_set_var` 工具将提取出的元数据存入变量 `calibrator_agent_result`，以供总控和其他智能体调用。存入的必须是一个纯净的 Python 字典对象（dict），格式如下：
        {
            "core_metrics": ["核心评价指标1", "指标2"],
            "dominant_paradigms": ["当前主导流派1", "流派2"],
            "critical_bottleneck": "最大的物理/工程/商业限制",
            "domain_limiters": ["aerospace", "aviation"]
        }
        4. 不要废话！通过 final_answer 提交第三步的变量。
        """
    )

    forager_agent = CustomAgent(
        model=model,
        tools=[SetVariableTool(), GetVariableTool()],
        name="forager_agent",
        description="基于领域锚点，将宏观主题拆解为四维检索词。必须传入 'task' (主题) 和 'context' (标定字典)。",
        instructions="""你是一个高级情报检索专家。请结合用户的原主题和传入的 context，生成 8 个正交的英文检索 Query。
        必须严格覆盖以下四个通用维度（每个维度2个Query）：
        1. 历史范式转移 (Paradigm Shifts)
        2. context 中提及的 dominant_paradigms 的最新突破
        3. context 中提及的 critical_bottleneck 的失效分析与妥协方案
        4. 试图颠覆 context 中 core_metrics 的前沿黑天鹅技术
        【防漂移强制红线】：你生成的每一个 Query 都【必须】包含领域限定后缀！
        具体做法：从 context 中提取 `domain_limiters`，并在每个 Query 末尾加上 `AND (limiter1 OR limiter2)`。
        使用`tool_set_var`工具将生成的 8 个 Query 存入变量 `forager_agent_result`，格式必须是一个纯正的 Python List[str] 对象.
        5. 不要废话！通过 final_answer 提交第四步的变量。
        """
    )

    analyst_agent = CustomAgent(
        model=model,
        tools=[ParseJsonTool(), SetVariableTool(), GetVariableTool()],
        name="analyst_agent",
        description="用于分析聚类后的情报数据。必须传入 'task' 参数。",
        instructions="""你是一个行业分析师。你将接收到文献聚类数据。请为每个聚类写一段 100 字的核心洞察。要求如下：
            1. 直接分析输入数据即可，绝不要调用任何外部工具或编写爬虫代码。
            2. 在输出完主题和瓶颈后，你必须作为裁判评估这些文献是否足够支撑写出一份深度大纲。
            3. 如果文献充分且覆盖了核心维度，请在最后单起一行输出：STATUS: PASS。
            4. 如果文献严重缺失某个关键维度，请在最后单起一行输出：STATUS: FAIL | MISSING: [用英文写出需要补充检索的具体关键词]。
            5. 使用 `tool_set_var` 工具将你的完整分析文本存入变量 `analyst_agent_result`。最后通过 final_answer 提交任务完成的提示。""",
        additional_authorized_imports=["json", "collections"]
    )

    outliner_agent = CustomAgent(
        model=model,
        tools=[ParseJsonTool(), SetVariableTool(), GetVariableTool()],
        name="outliner_agent",
        description="基于洞察生成具有因果递进逻辑的三级大纲树。必须传入 'task' 参数。",
        instructions="""你是一个顶级综述架构师。请基于输入的洞察数据，构建深度调研的三级大纲。
        【逻辑红线】：大纲章节之间必须呈现强烈的因果递进关系（从旧范式 -> 当前主流 -> 痛点 -> 前沿破局点）。
        【主键引用红线 (CRITICAL)】：每一个最底层的章节节点（如 level_2），必须包含一个 `supporting_papers` 字段。
        该字段是一个严格的列表，里面 **只允许存放从 context 中提取的文献 id**！
        🚀【严禁偷懒 (CRITICAL)】：
        1. 每一个底层小节的 `supporting_papers` 列表中，必须至少分配 3 到 6 篇文献 ID。
        2. 整份大纲引用的【不重复文献总数】绝对不能低于 raw_data 中文献数量的一半！
        3. 覆盖所有的洞察点！
        大纲必须是合法的JSON格式。在生成完大纲的时候，使用 `tool_set_var` 工具将大纲对象存入变量 `outliner_agent_result`，最后使用 `final_answer` 宣告完成。""",
        additional_authorized_imports=["json", "collections"]
    )

    # ==========================================
    # 3. 实例化第二阶段智能体群 (Writing Matrix)
    # ==========================================
    section_writer_agent = CustomAgent(
        model=model,
        tools=[], 
        name="section_writer_agent",
        description="学术长文底层的纯粹主笔节点。",
        instructions="""你是一名世界级的科技领域首席研究员。
        【任务】：根据传入的标题、核心论点和 RAG 语料，撰写学术报告的一个小节。
        【写作红线】：
        1. 严禁无营养废话。所有数据和结论必须直接来源于语料。
        2. 强制学术引用：必须在句末对所有使用的数据和观点进行文献引用。
           - 若语料来源为本地文献，使用 [paper_id]。
           - 若语料来源为在线网页，直接使用该 URL 作为引用标号。
        3. 绝对禁止输出“[注: 无本地文献支撑]”这类免责声明！
        4. 严禁尝试调用 visit_webpage 等网页浏览工具！所有网页的内容系统已经提取在语料中。
        5. 格式要求：你必须且只能使用 python 代码块来提交结果，即 `final_answer("你的正文字符串")`，严禁直接输出纯文本。
        """
    )

    # ==========================================
    # 4. 组建两个中层大管家 (Director Agents)
    # ==========================================
    skills = load_skills_from_directory("skills", model=model)

    # 4.1 Research 大管家
    research_tools = [AcademicSearchTool(), InsightExtractorTool(), SemanticClusterTool(), PaperDownloaderTool(), SaveFileTool(), ParseJsonTool(), SetVariableTool(), GetVariableTool()]
    research_tools.extend(skills)

    research_director_agent = CodeAgent(
        name="research_director_agent",
        description="第一阶段总管：负责深度研究大纲规划与调度。传入包含 'topic' 和 'run_dir' 的指令启动。它会自动调度四大分析智能体并最终在目录中生成 deep_research_outline.json 文件。",
        tools=research_tools,
        managed_agents=[calibrator_agent, forager_agent, analyst_agent, outliner_agent],
        model=model,
        instructions="""你是一个科研前研架构师。配备了4个下级专家智能体。
        接到指令后，请：
        调用 `deep_research_director_sop` 工具，严格按照里面的 6 个阶段步骤调度下级智能体。
        跑完流程后，通过 `final_answer` 向你的上级汇报“大纲生成完毕”。""",
        additional_authorized_imports=["json", "time", "ast", "re", "os"]
    )

    # 4.2 Write 大管家
    rag_tool = UniversalRAGTool()
    assembler_tool = ReportAssemblerTool(writer_agent=section_writer_agent, rag_tool=rag_tool)
    
    writer_tools = [ParseJsonTool(), FlattenOutlineTool(), assembler_tool, SaveFileTool(), SetVariableTool(), GetVariableTool(), GenerateBibliographyTool(), LoadFileTool(), GenerateAbstractTool(), GenerateConclusionTool(), MdToWordTool()]
    writer_tools.extend(skills)

    writer_director_agent = CodeAgent(
        name="writer_director_agent",
        description="第二阶段总管：负责长文自动撰写。传入包含 'run_dir' 的指令启动。它会读取 outline 文件并自动调度底层节点组装全篇万字报告。",
        tools=writer_tools,
        model=model,
        instructions="""你是一个长文写作总架构师。
        请从long_writer_section_sop获取写作指南，并严格按照指南要求执行代码，完成学术报告的撰写。
        之后立即通过 `final_answer` 向你的上级汇报“长文写作完毕”。
        """,
        additional_authorized_imports=["json", "os"]
    )

    # ==========================================
    # 5. 实例化顶层综述智能体 (The Review Master)
    # ==========================================
    review_agent = CodeAgent(
        name="review_agent",
        description="顶级综述主控引擎。端到端协调 Research 和 Write 两个阶段。",
        tools=[SetVariableTool(), GetVariableTool()],
        managed_agents=[research_director_agent, writer_director_agent], # ✨ 核心：将原有的两大主角变成了工具池里的组件
        model=model,
        instructions="""你是一个顶级的全自动综述生成主控智能体 (Review Agent)。你的任务是统筹调度手底下的两个总管智能体，完成从 0 到 1 的综述写作流水线。
        请严格按照以下顺序编写并执行 Python 代码：
        1. 呼叫 `research_director_agent`，命令它执行深度研究并生成大纲。需要将目标主题和运行目录告诉它。
        2. 等待 `research_director_agent` 成功返回后，立马呼叫 `writer_director_agent`，命令它读取生成的大纲并执行全篇学术长文的撰写和组装。同样需要将运行目录告诉它。
        3. 等待所有撰写任务结束后，使用 `final_answer` 向用户输出全流程成功的总结。
        """,
        additional_authorized_imports=["os", "json"]
    )

    # ==========================================
    # 6. 状态注入与任务启动
    # ==========================================
    target_topic = "Automated Scientific Discovery with Large Language Models"
    
    # 将共享上下文信息提前注入给所有层级的引擎，防止丢失
    research_director_agent.state["target_topic"] = target_topic
    research_director_agent.state["run_dir"] = run_dir
    writer_director_agent.state["run_dir"] = run_dir
    review_agent.state["target_topic"] = target_topic
    review_agent.state["run_dir"] = run_dir

    try:
        smol_logger.info("🚀 Review Agent 全流程主控引擎启动...")
        
        # 向顶层 Agent 下达启动口令
        task_prompt = f"请启动全流程综述撰写任务。目标主题是：'{target_topic}'。工作目录是：'{run_dir}'。请依次调度 research_director_agent 和 writer_director_agent 完成任务。"
        review_agent.run(task_prompt)
        
        print(f"\n🎉 端到端全流程执行完成！所有大纲、过程日志及最终的万字综述产出物均已落盘至: {run_dir}")
    except Exception as e:
        smol_logger.error(f"❌ 发生致命错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()