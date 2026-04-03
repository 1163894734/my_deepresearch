import argparse
import os
import threading
import time
from scripts.test_identification_agent_simple import get_highly_realistic_docs
from smolagents import InferenceClientModel
from smolagents import OpenAIModel

from dotenv import load_dotenv
from scripts.text_inspector_tool import TextInspectorTool
from scripts.custom_tools import LongWriterTool
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
    CustomAgent,
)
from scripts.skill_loader import load_skills_from_directory
from scripts.long_writer_agent_v3 import LongWriterAgent
from scripts.identification_agent import IdentificationAgent
from scripts.interpretation_agent import InterpretationAgent


load_dotenv(override=True)

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
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
        api_key=os.environ["DYM_API_KEY"],
        max_tokens=8192
    )
    skills_dir = os.path.join(base_dir, "skills")
    
    print(f"Loading skills from {skills_dir}...")
    custom_tools = load_skills_from_directory(skills_dir, model=model)
    
    print(f"Loaded {len(custom_tools)} custom tools: {[t.name for t in custom_tools]}")
    # long_writer_tool = LongWriterTool(model)
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
    text_limit = 20000
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

    # 为 LongWriterAgent 添加 web_search 工具
    web_search_tool = DuckDuckGoSearchTool()
    web_search_tool.name = "web_search"  # 确保名称匹配
    
    # 创建 LongWriterAgent V3（多阶段长文本生成代理）
    long_writer_agent = LongWriterAgent(
        model=model,
        tools=custom_tools + [web_search_tool],  # 传入所有 skills + web_search
        managed_agents=[text_webbrowser_agent],  # 细粒度检索时可调用 search_agent
        max_steps=30,
        verbosity_level=2,
        name="long_writer_agent",
        description="""**专门负责长文本学术写作的智能体（论文、综述、调研报告）**

**何时必须委托给我**：
- 用户要求写"综述"、"调研报告"、"学术论文"、"技术报告"
- 要求字数超过 5000 字或 1 万字
- 需要"参考文献"、"引用文献"、"列出文献"
- 需要多章节结构化内容（引言、正文、结论等）
- 要求"完整"、"详尽"、"系统性"的文档

**我的核心能力**：
- 自动生成学术大纲 + 迭代优化
- 分段撰写 + 证据引用管理
- 反思循环确保质量
- 自动整理参考文献列表

**委托方式**：
直接调用 `long_writer_agent(task="用户的完整任务描述")`，不要自己调用 outline_generation、section_write 等工具。
""",
        provide_run_summary=True,
    )

    # 创建 IdentificationAgent（前沿技术识别代理）
    identification_agent = IdentificationAgent(
        model=model,
        tools=custom_tools + [web_search_tool],
        max_steps=20,
        verbosity_level=2,
        name="identification_agent",
        description="""进行前沿技术识别时使用。

适用场景：
- 输入大量带时间戳文档，需要筛选是否属于某技术领域
- 需要聚类识别技术流派/细分方向
- 需要按年度分析增长、占比与技术突现（burst）
- 需要多专家投票后输出候选前沿技术
""",
        provide_run_summary=True,
    )

    # 创建 InterpretationAgent（技术名词解读代理）
    interpretation_agent = InterpretationAgent(
        model=model,
        tools=custom_tools + [web_search_tool],
        max_steps=20,
        verbosity_level=2,
        name="interpretation_agent",
        description="""进行技术名词解读、术语解释与多风格改写时使用。

适用场景：
- 从用户输入中抽取核心技术名词
- 并行检索本地向量库与Web资料并去重重排
- 生成客观事实底稿
- 输出百科版、专报版、科普版三种风格解读
""",
        provide_run_summary=True,
    )

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, text_limit)] + custom_tools,
        max_steps=20,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=2,
        # 将 long_writer_agent 和 search_agent 注册为 managed agents
        managed_agents=[
            long_writer_agent,
            identification_agent,
            interpretation_agent,
            text_webbrowser_agent,
        ],

    )

    return manager_agent


def main():
    args = parse_args()

    agent = create_agent(model_id="ollama/qwen2.5:1.5b")

    test =""""""
    answer = agent.run(args.question)

    print(f"Got this answer: {answer}")


if __name__ == "__main__":
    main()
