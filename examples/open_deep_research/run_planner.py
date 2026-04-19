import argparse
import json
import os
from dotenv import load_dotenv

from smolagents import OpenAIModel, CustomAgent, CodeAgent, DuckDuckGoSearchTool, Tool
from scripts.skill_loader import load_skills_from_directory
from scripts.long_writer_agent_v3 import LongWriterAgent, LongWriterRuntimeState
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.citation_validator import CitationValidator
from utils import common_utils

load_dotenv(override=True)

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


class CitationValidationTool(Tool):
    name = "citation_validation"
    description = "验证段落中的引用并返回校正后的引用库、修订后的段落和验证日志。"
    inputs = {
        "citations": {"type": "object", "description": "当前引用库，字典格式"},
        "paragraph": {"type": "string", "description": "待验证的段落文本"},
        "section_ref": {"type": "string", "description": "当前段落编号或标题", "nullable": True},
    }
    output_type = "string"

    def __init__(self, model):
        super().__init__()
        self.validator = CitationValidator(model=model, remove_entire_invalid_sentence=False)

    def forward(self, citations, paragraph: str, section_ref: str = "") -> str:
        updated_citations, validated_paragraph, step_logs = self.validator.run_five_step_validation(
            citations,
            paragraph,
            section_ref,
        )
        return json.dumps(
            {
                "available_citations": updated_citations,
                "paragraph": validated_paragraph,
                "step_logs": step_logs,
            },
            ensure_ascii=False,
        )


PLANNING_INSTRUCTIONS = """
你是长文本报告的规划智能体。
任务开始时先调用 `get_planning_sop` 读取规划 SOP，然后严格按 SOP 执行。
你只负责输出规划结果，不写正文。
最终输出必须是 JSON，对象至少包含 `outline`、`available_citations`、`planning_metadata`。
"""


def build_web_browser_agent(model):
    text_limit = 20000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    web_tools = [
        DuckDuckGoSearchTool(),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]

    text_webbrowser_agent = CustomAgent(
        model=model,
        tools=web_tools,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="custom_search_agent",
        description="A team member that will search the internet to answer your question.",
        provide_run_summary=True,
    )

    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""
    return text_webbrowser_agent





# section_agent = CodeAgent(
#     model=model,
#     tools=shared_tools,
#     managed_agents=[text_webbrowser_agent],
#     max_steps=28,
#     verbosity_level=2,
#     planning_interval=4,
#     name="section_writer_agent",
#     description="负责长文本报告的章节撰写",
#     instructions=SECTION_INSTRUCTIONS,
#     provide_run_summary=True,
#     additional_authorized_imports=["json", "time", "re"],
# )

# finalization_agent = CodeAgent(
#     model=model,
#     tools=shared_tools,
#     managed_agents=[text_webbrowser_agent],
#     max_steps=20,
#     verbosity_level=2,
#     planning_interval=4,
#     name="finalization_agent",
#     description="负责长文本报告的摘要、参考文献和最终排版",
#     instructions=FINALIZATION_INSTRUCTIONS,
#     provide_run_summary=True,
#     additional_authorized_imports=["json", "time", "re"],
# )



def parse_args():
    parser = argparse.ArgumentParser(description="独立运行 LongWriterAgent")
    parser.add_argument(
        "--question", 
        type=str, 
        nargs="?",
        help="例如: '写一篇关于大语言模型在多模态方向最新进展的学术综述，要求包含参考文献。'",
        default="介绍一下大模型领域的主要文献和近期进展，形成一篇完整、内容详尽的中文调研报告，整体字数要求大于2w字"
    )
    return parser.parse_args()


def agent_run():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 初始化模型 (保持你原有的模型配置)
    model = common_utils.ModelProvider.get_model()
    
    # 2. 从本地加载 Skills (自定义工具)
    skills_dir = os.path.join(base_dir, "skills")
    print(f"正在从 {skills_dir} 加载 Skills...")
    custom_tools = load_skills_from_directory(skills_dir, model=model)
    print(f"成功加载 {len(custom_tools)} 个自定义工具: {[t.name for t in custom_tools]}")

    # 3. 初始化验证工具、浏览智能体与三个阶段智能体
    citation_validation_tool = CitationValidationTool(model=model)
    text_webbrowser_agent = build_web_browser_agent(model)

    web_search_tool = DuckDuckGoSearchTool()
    web_search_tool.name = "web_search"
    shared_tools = custom_tools + [web_search_tool, citation_validation_tool]
    

    writer_state = LongWriterRuntimeState()


    GENERIC_INSTRUCTIONS = """
    你是一个高度自主的 AI 助手。你可以编写并在沙盒中运行 Python 代码来处理数据。
    当你遇到复杂的垂直业务请求时，请先检查工具列表中是否有对应的 SOP 或指南工具，并优先调用它们来了解你应该怎么做。
    """
    # 过滤 academic_search_agent 的工具列表，只保留指定名字的工具
    academic_search_tools = [t for t in shared_tools if t.name in ["final_answer", "academic_search", "json_load"]]
    
    academic_search_agent = CodeAgent(
        model=model,
        tools=academic_search_tools,
        managed_agents=[text_webbrowser_agent],
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="academic_search_agent",
        description="负责通过用户的查询进行学术文献的检索",
        instructions=GENERIC_INSTRUCTIONS,
        additional_authorized_imports=["json", "time", "re"],
    )

    GENERIC_INSTRUCTIONS2 = """
    你是一个高度自主的 AI 助手。你可以编写并在沙盒中运行 Python 代码来处理数据。
    你解决的是通过关键词扩展进行文献检索的问题，请先检查工具列表中是否有对应的 SOP 或指南工具，并优先调用它们来了解你应该怎么做。
    """

    keywords_search_agent = CustomAgent(
        model=model,
        tools=shared_tools,
        max_steps=20,
        verbosity_level=2,
        name="keywords_search_agent",
        description="负责关键词扩展检索",
        instructions=GENERIC_INSTRUCTIONS2,
        additional_authorized_imports=["json", "time", "re"],
    )

    planning_agent = CustomAgent(
        model=model,
        tools=shared_tools,
        managed_agents=[text_webbrowser_agent]+[academic_search_agent]+[keywords_search_agent],
        max_steps=24,
        verbosity_level=2,
        name="planning_agent",
        description="负责长文本写作的规划与大纲生成",
        instructions=PLANNING_INSTRUCTIONS,
        provide_run_summary=True,
        additional_authorized_imports=["json", "time", "re"],
    )

    return planning_agent


def main():
    args = parse_args()

    print(f"正在初始化 planning_agent...")
    agent = agent_run()

    print(f"\n🚀 开始执行任务: {args.question}\n")
    # 直接调用 Agent 的 run 方法
    answer = agent.run(args.question)

    print(f"\n✅ 任务执行完毕，最终输出如下:\n")
    print(answer)


if __name__ == "__main__":
    main()