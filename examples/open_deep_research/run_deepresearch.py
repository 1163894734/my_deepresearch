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
    calibrator_agent = CustomAgent(
        model=model,
        tools=[AcademicSearchTool(),SetVariableTool(),GetVariableTool()], 
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
            "critical_bottleneck": "最大的物理/工程/商业限制"
            "domain_limiters": ["aerospace", "aviation"]  # <=== 新增：提取2-3个该领域专属的排他性英文限定词，用于防止跨领域检索污染
        }
        4. 不要废话！通过 final_answer 提交的第三步的变量。
        """
    )

    # 2.1 觅食者智能体 (Forager) - 升级为“四维通用觅食”
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
        具体做法：从 context 中提取 `domain_limiters`，并在每个 Query 末尾加上 `AND (limiter1 OR limiter2)`。
        例如：`historical paradigm shifts in composite materials AND (aerospace OR aviation)`。
        使用``tool_set_var``工具将生成的 8 个 Query 存入变量 `forager_agent_result`，格式必须是一个纯正的 Python List[str] 对象.
        5. 不要废话！通过 final_answer 提交的第四步的变量。
        """
    )

    # 2.2 分析师智能体 (Analyst) - 保持不变
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

    # 2.3 编排者智能体 (Outliner) - 升级为“通用因果编排”
    # 2.3 编排者智能体 (Outliner) - 升级为“极简主键引用”架构
    outliner_agent = CustomAgent(
        model=model,
        tools=[ParseJsonTool(),SetVariableTool(),GetVariableTool()],
        name="outliner_agent",
        description="基于洞察生成具有因果递进逻辑的三级大纲树。必须传入 'task' 参数。",
        instructions="""你是一个顶级综述架构师。请基于输入的洞察数据，构建深度调研的三级大纲。
        【逻辑红线】：大纲章节之间必须呈现强烈的因果递进关系（从旧范式 -> 当前主流 -> 痛点 -> 前沿破局点）。
        【主键引用红线 (CRITICAL)】：每一个最底层的章节节点（如 level_2），必须包含一个 `supporting_papers` 字段。
        该字段是一个严格的列表，里面 **只允许存放从 context 中提取的文献 id (例如 ["https://openalex.org/W123...", "paper_456"])**！
        绝对严禁在列表里存放字典、URL路径或作者名！所有的元数据都在外部数据库中，大纲只负责存 ID 建立映射。
        🚀【严禁偷懒 (CRITICAL)】：
        作为深度研究报告，文献引用必须丰满！你必须尽可能穷尽式地利用输入的 raw_data。
        1. 每一个底层小节的 `supporting_papers` 列表中，必须至少分配 3 到 6 篇文献 ID。
        2. 整份大纲引用的【不重复文献总数】绝对不能低于 raw_data。中文献数量的一半！
        3. 仔细审查每一篇文献的洞察，把它们分类塞进最合适的章节中，不要只挑几篇代表作。
        4. 覆盖所有的洞察点！每一个洞察都必须在大纲中找到它的归宿，绝不能有遗漏！
        期望的节点格式范例：
        {
            "chapter_title": "1.1 纳米尺度精准调控",
            "core_argument": "纳米材料实现靶向递送，但面临毒性挑战",
            "supporting_papers": [
                "https://openalex.org/W3126951392",
                "https://openalex.org/W2620160911"
            ]
        }
        大纲必须是合法的JSON格式。在生成完大纲的时候，使用 `tool_set_var` 工具将大纲对象存入变量 `outliner_agent_result`，
        最后使用 `final_answer` 宣告完成。""",
        additional_authorized_imports=["json", "collections"]
    )
    # =========================
    # 3. 实例化底层工具与总控
    # =========================
    tools = [AcademicSearchTool(), InsightExtractorTool(), SemanticClusterTool(), PaperDownloaderTool(), SaveFileTool(), ParseJsonTool(),SetVariableTool(),GetVariableTool()]
    skills = load_skills_from_directory("skills", model=model)
    tools.extend(skills)

    director_agent = CodeAgent(
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

    # 测试通用性：这里可以换成任何陌生领域
    target_topic = "Automated Scientific Discovery with Large Language Models"
    director_agent.state["target_topic"] = target_topic
    director_agent.state["run_dir"] = run_dir

    try:
        smol_logger.info("🎬 Director Agent 启动通用深度研究引擎...")
        director_agent.run("请根据注入的 target_topic 变量，启动深度研究大纲规划。")
        print(f"\n🎉 规划完成！通用架构大纲已落盘至: {run_dir}/deep_research_outline.json")
    except Exception as e:
        smol_logger.error(f"❌ 发生致命错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()