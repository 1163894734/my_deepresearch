import argparse
import os
from dotenv import load_dotenv

from smolagents import OpenAIModel, CustomAgent, DuckDuckGoSearchTool
from scripts.skill_loader import load_skills_from_directory
from scripts.long_writer_agent_v3 import LongWriterAgent
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
from utils import common_utils

# 加载环境变量 (DYM_API_KEY, SERPAPI_API_KEY 等)
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


def create_long_writer_agent():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 初始化模型 (保持你原有的模型配置)
    model = common_utils.ModelProvider.get_model()
    
    # 2. 从本地加载 Skills (自定义工具)
    skills_dir = os.path.join(base_dir, "skills")
    print(f"正在从 {skills_dir} 加载 Skills...")
    custom_tools = load_skills_from_directory(skills_dir, model=model)
    print(f"成功加载 {len(custom_tools)} 个自定义工具: {[t.name for t in custom_tools]}")

    # 3. 构建供 LongWriterAgent 调用的底层网络浏览智能体 (text_webbrowser_agent)
    text_limit = 20000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    WEB_TOOLS = [
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
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="custom_search_agent",
        description="""A team member that will search the internet to answer your question.
        Ask him for all your questions that require browsing the web.
        Provide him as much context as possible, in particular if you need to search on a specific timeframe!
        And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
        Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
        """,
        provide_run_summary=True,
    )
    
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    # 4. 初始化全局搜索工具
    web_search_tool = DuckDuckGoSearchTool()
    web_search_tool.name = "web_search"
    
    # 5. 实例化最终的 LongWriterAgent
    long_writer_agent = LongWriterAgent(
        model=model,
        tools=custom_tools + [web_search_tool],       # 传入所有的 skills 以及 web_search
        managed_agents=[text_webbrowser_agent],       # 将网页浏览智能体作为子智能体挂载
        max_steps=30,
        verbosity_level=2,
        name="long_writer_agent",
        description="专门负责长文本学术写作的智能体（论文、综述、调研报告）",
        provide_run_summary=True,
    )

    return long_writer_agent


def main():
    args = parse_args()

    print("正在初始化 LongWriterAgent...")
    agent = create_long_writer_agent()

    print(f"\n🚀 开始执行任务: {args.question}\n")
    # 直接调用 LongWriterAgent 的 run 方法
    answer = agent.run(args.question)

    print(f"\n✅ 任务执行完毕，最终输出如下:\n")
    print(answer)


if __name__ == "__main__":
    main()