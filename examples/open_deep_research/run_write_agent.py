import os
import json
import logging
from smolagents import CodeAgent, CustomAgent
# 引入所有所需工具
from scripts.custom_tools import AcademicRAGTool, GenerateBibliographyTool, GetVariableTool, SaveFileTool, ParseJsonTool, FlattenOutlineTool, ReportAssemblerTool, SetVariableTool, UniversalRAGTool, LoadFileTool
import utils.common_utils as common_utils

# =========================
# 1. 基础配置与日志
# =========================
base_dir = os.path.dirname(os.path.abspath(__file__))
# ⚠️ 注意替换目录
run_dir = "/Users/wangchao/project/my_deepresearch/examples/outputs/deep_research_20260423_101758"

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

    # =========================
    # 3. 实例化最高总管
    # =========================
    writer_director_agent = CodeAgent(
        name="writer_director_agent",
        description="最高总管。负责解析大纲并调用底层的自动组装引擎。",
        tools=[ParseJsonTool(), FlattenOutlineTool(), assembler_tool, SaveFileTool(),SetVariableTool(),GetVariableTool(), GenerateBibliographyTool(), LoadFileTool()],
        model=model,
        additional_authorized_imports=["json"],
        instructions="""你是一个长文写作总架构师 (CodeAgent)。
        请在代码沙盒中严格执行以下 **3行代码**，不要做任何多余的修改和循环！

        ```python
        # 1. 展平大纲并存入共享变量
        使用load_file(run_dir+"/final_papers.json", as_json=True)工具读取final_papers文件内容，并将读取到的内容使用tool_set_var存入全局变量"PAPER_DB"。
        flat_data = tool_flatten_outline(outline_data=tool_parse_json(text=outline_str))
        tool_set_var(key="flat_data", value=flat_data)
        
        # 2. 调用自动化装配引擎 (它在底层帮你完成了所有的循环、RAG和写作调度)
        full_report = tool_assemble_report(main_title=flat_data['main_title'], tasks=flat_data['tasks'])
        save_file(content=full_report, out_dir=run_dir, file_name="final_academic_report", out_type="md")
        
        
        # 3. 归档落盘
        formatted_content = tool_generate_bibliography(report_content=full_report, tasks_data=flat_data['tasks'])
        save_file(content=formatted_content, out_dir=run_dir, file_name="final_academic_report_finale", out_type="md")

        final_answer("全篇万字学术报告已完美撰写并保存！")
        ```
        【注意】
        环境中已经注入了变量 `outline_str`（大纲字符串）和 `run_dir`（当前运行目录），请直接使用，不要通过工具获取这两个变量的值。
        你需要做的就是按照上面提供的代码框架，调用工具完成写作任务。
        所有复杂的循环、RAG、调度逻辑都已经被封装在了 `tool_assemble_report` 这个自动化装配引擎中，你只需要正确传入参数即可。
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