import argparse
import os
import threading
import time
from smolagents import InferenceClientModel
from smolagents import OpenAIModel

from dotenv import load_dotenv
from huggingface_hub import login
from scripts.text_inspector_tool import TextInspectorTool
from scripts.long_writer_tool import LongWriterTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    DuckDuckGoSearchTool,
    # InferenceClientModel,
    LiteLLMModel,
    ToolCallingAgent,
)


load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question", type=str, help="for example: 'How many studio albums did Mercedes Sosa release before 2007?'"
    )
    parser.add_argument("--model-id", type=str, default="o1")
    return parser.parse_args()


custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

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


def create_agent(model_id="o1"):
    model_params = {
        "model_id": model_id,
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": 8192,
    }
    if model_id == "o1":
        model_params["reasoning_effort"] = "high"
    # model = LiteLLMModel(**model_params)

    model = OpenAIModel(
        model_id="Qwen3-235B-A22B-Instruct-2507",
        api_base="https://llmapi.paratera.com/v1",
        api_key=os.environ["DYM_API_KEY"]
    )

    long_writer_tool = LongWriterTool(model)
    # model = OpenAIModel(
    #     model_id="deepseek-chat",  # 根据DeepSeek V3的实际模型ID调整
    #     api_base="https://api.deepseek.com/v1",  # 替换为DeepSeek V3的API基础地址
    #     api_key=os.environ["DEEPSEEK_API_KEY"],  # 需设置环境变量存储API密钥
    # )

    # model = OpenAIModel(
    #     model_id="gpt-5",  # 根据DeepSeek V3的实际模型ID调整
    #     api_base="https://api.gptsapi.net/v1",  # 替换为DeepSeek V3的API基础地址
    #     api_key=os.environ["GPTSAPI_KEY"],  # 需设置环境变量存储API密钥
    #     # max_tokens=2048,  # 强制必填：设置生成的最大 tokens 数（根据 GPTs 限制调整）
    #     # temperature=0.7,  # 可选：保持原有配置
    # )
    text_limit = 100000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    WEB_TOOLS = [
        # GoogleSearchTool(provider="serper"),
        DuckDuckGoSearchTool(),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
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

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, text_limit), long_writer_tool],
        max_steps=20,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=2,
        managed_agents=[text_webbrowser_agent],

    )

    return manager_agent


def main():
    args = parse_args()

    agent = create_agent(model_id="ollama/qwen2.5:1.5b")

    answer = agent.run(args.question)

    print(f"Got this answer: {answer}")


if __name__ == "__main__":
    main()
