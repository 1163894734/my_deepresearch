import os
import json
import logging
from smolagents import CodeAgent, CustomAgent
from scripts.skill_loader import load_skills_from_directory
# 引入所有所需工具
from scripts.custom_tools import AcademicRAGTool, GenerateAbstractTool, GenerateBibliographyTool, GenerateConclusionTool, GetVariableTool, SaveFileTool, ParseJsonTool, FlattenOutlineTool, ReportAssemblerTool, SetVariableTool, UniversalRAGTool, LoadFileTool, MdToWordTool
import utils.common_utils as common_utils

# =========================
# 1. 基础配置与日志
# =========================
base_dir = os.path.dirname(os.path.abspath(__file__))
# ⚠️ 注意替换目录
run_dir = "/Users/wangchao/project/my_deepresearch/examples/outputs/deep_research_20260509_164615"

smol_logger = logging.getLogger("writer_agent")
smol_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
smol_logger.addHandler(console_handler)
file_handler = logging.FileHandler(os.path.join(run_dir, "writer_run.log"), encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
smol_logger.addHandler(file_handler)

def main():
    smol_logger.info(f"📁 启动【防偷懒版】全智能体写作流水线: {run_dir}")
    model = common_utils.ModelProvider.get_model()

    outline_path = os.path.join(run_dir, "deep_research_outline.json")
    if not os.path.exists(outline_path):
        smol_logger.error(f"❌ 找不到大纲文件: {outline_path}")
        return
        
    with open(outline_path, 'r', encoding='utf-8') as f:
        outline_str = f.read()

    # 实例化需要透传的工具


    # =========================
    # 2. 实例化下级主笔专家
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

    # rag_tool = AcademicRAGTool()
    rag_tool = UniversalRAGTool()
    assembler_tool = ReportAssemblerTool(writer_agent=section_writer_agent, rag_tool=rag_tool)
    tools = [ParseJsonTool(), FlattenOutlineTool(), assembler_tool, SaveFileTool(),SetVariableTool(),GetVariableTool(), GenerateBibliographyTool(), LoadFileTool(), GenerateAbstractTool(), GenerateConclusionTool(), MdToWordTool()]
    skills = load_skills_from_directory("skills", model=model)
    tools.extend(skills)
    # =========================
    # 3. 实例化最高总管
    # =========================
    writer_director_agent = CodeAgent(
        name="writer_director_agent",
        description="最高总管。负责解析大纲并调用底层的自动组装引擎。",
        tools=tools,
        model=model,
        additional_authorized_imports=["json"],
        instructions="""你是一个长文写作总架构师 (CodeAgent)。
        请从long_writer_section_sop获取写作指南，并严格按照指南要求执行代码，完成学术报告的撰写。
        """
    )

    writer_director_agent.state["outline_str"] = outline_str
    writer_director_agent.state["run_dir"] = run_dir

    try:
        smol_logger.info("🎬 Writer Director Agent 启动...")
        writer_director_agent.run("请执行指令中的代码，完成长文的撰写与保存。")
    except Exception as e:
        smol_logger.error(f"❌ 发生致命错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()