import argparse
import os
import threading
import time
import json  # 用于解析和格式化输出
from smolagents import InferenceClientModel
from smolagents import OpenAIModel

from dotenv import load_dotenv
from scripts.text_inspector_tool import TextInspectorTool
# from scripts.custom_tools import LongWriterTool # 如果你不需要可以注释
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
    LiteLLMModel,
)
# 假设你本地有 CustomAgent，如果没有请替换为 ToolCallingAgent
from smolagents import ToolCallingAgent as CustomAgent 

from scripts.skill_loader import load_skills_from_directory
from scripts.long_writer_agent_v3 import LongWriterAgent
from scripts.identification_agent import IdentificationAgent
from scripts.interpretation_agent import InterpretationAgent

load_dotenv(override=True)

append_answer_lock = threading.Lock()

paper_number = 5


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

    # 使用你配置的 Qwen 模型
    model = OpenAIModel(
        model_id="Qwen3-235B-A22B-Instruct-2507",
        api_base="https://llmapi.paratera.com/v1",
        api_key=os.environ["DYM_API_KEY"]
    )
    skills_dir = os.path.join(base_dir, "skills")
    
    print(f"Loading skills from {skills_dir}...")
    custom_tools = load_skills_from_directory(skills_dir, model=model)
    print(f"Loaded {len(custom_tools)} custom tools: {[t.name for t in custom_tools]}")
    
    text_limit = 100000
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

    # 为 LongWriterAgent 添加 web_search 工具
    web_search_tool = DuckDuckGoSearchTool()
    web_search_tool.name = "web_search"  
    
    # 创建 LongWriterAgent V3
    long_writer_agent = LongWriterAgent(
        model=model,
        tools=custom_tools + [web_search_tool],
        managed_agents=[text_webbrowser_agent],
        max_steps=30,
        verbosity_level=2,
        name="long_writer_agent",
        description="""专门负责长文本学术写作的智能体（论文、综述、调研报告）。直接调用 `long_writer_agent(task="...")`。""",
        provide_run_summary=True,
    )

    # 创建 IdentificationAgent
    identification_agent = IdentificationAgent(
        model=model,
        tools=custom_tools + [web_search_tool],
        max_steps=20,
        verbosity_level=2,
        name="identification_agent",
        description="""进行前沿技术识别时使用。""",
        provide_run_summary=True,
    )

    # 创建 InterpretationAgent
    interpretation_agent = InterpretationAgent(
        model=model,
        tools=custom_tools + [web_search_tool],
        max_steps=20,
        verbosity_level=2,
        name="interpretation_agent",
        description="""进行技术名词解读、术语解释与多风格改写时使用。""",
        provide_run_summary=True,
    )

    # 主控 Agent：寻找 20 篇文献步骤多，max_steps 设为 30
    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, text_limit)] + custom_tools,
        max_steps=30, 
        verbosity_level=2,
        additional_authorized_imports=["*", "json", "time"], 
        planning_interval=2,
        managed_agents=[
            long_writer_agent,
            identification_agent,
            interpretation_agent,
            text_webbrowser_agent,
        ],
    )

    return manager_agent

def search(question: str):

    # 创建主控 Agent (此时使用的是自定义里的 Qwen3 配置)
    agent = create_agent(model_id="ollama/qwen2.5:1.5b")

# ====== 核心改动：加入正反例约束，并将 keywords 改为 technical_terms ======
    task_prompt = f"""
    你的核心任务是围绕主题："{question}" 进行深度互联网检索。

    【工作流说明】
    1. 你需要调用 `search_agent` 在互联网上搜索并阅读网页、文献或技术报告。
    2. 你必须收集到整整 **{paper_number}篇** 高质量的最相关文献/资料。
    3. 在你的 Python 代码执行环境中，你需要建立一个列表（list），将每次找到的文献结构化为字典（dict）并存入列表中。
    4. 为了防止中途意外中断，你可以在每次找到几篇文献后，使用 `with open('temp_papers.json', 'w')` 将目前收集到的 list 写入本地文件暂存。

    【关于技术名词（technical_terms）的极度严格要求】
    我需要你扮演资深的学术评审专家。在提取文献的 5 个技术名词时，**绝对禁止提取宽泛的、毫无信息量的废话**（如“人工智能”、“大模型”、“性能优化”、“分析方法”、“系统框架”等）。
    你必须提取这篇文献中最硬核的**专有技术名词、核心算法名称、特殊网络架构、或极度细分的研究方向**！
    ❌ 错误示范：["自然语言处理", "性能提升", "文本生成", "AI系统", "优化方法"]
    ✅ 正确示范：["MoE混合专家架构", "DPO直接偏好对齐", "FlashAttention机制", "RoPE旋转位置编码", "检索增强生成(RAG)"]

    【输出格式要求 - 极其重要】
    在收集满 {paper_number} 篇文献后，请你**必须**在你的 Python 代码中，按照以下结构构建一个完整的字典 `data`，然后使用 `json.dumps(data, ensure_ascii=False, indent=2)` 转换为 JSON 字符串，并通过 `final_answer(json_string)` 将其输出。

    你的 Python 字典必须具有如下的 key 结构：
    ```python
    data = {{
        "query": "{question}",
        "items": [
            {{
                "title": "文章标题",
                "url": "文章链接",
                "source_type": "webpage|paper|report|news|other",
                "time": "YYYY 或 YYYY-MM",
                "direction": "该研究所属的最细分的技术方向（例如：并非填大语言模型，而是填LLM推理加速）",
                "content_summary": "核心内容总结（不少于50字）",
                "technical_terms": ["专业名词1", "专业名词2", "专业名词3", "专业名词4", "专业名词5"], # 必须严格遵照上述正反例要求，提取 5 个硬核技术/算法名词
                "existing_problems": ["现存问题1", "现存问题2"],
                "foundation": "该研究或内容是建立在什么算法/理论基础上的",
                "outlook": "对该细分领域的未来展望",
                "future_directions": ["未来方向1", "未来方向2"],
                "relevance_score": 9, # 0-10整数，表示与检索主题的相关性
                "evidence_quote": "原文中能支撑上述总结的原话引用",
                "authors": "作者名称，如无则填 unknown",
                "year": "YYYY格式",
                "apa_citation": "(Author, Year)"
            }},
            # ... 此处省略，但你最终必须组装 {paper_number} 个这样的字典元素
        ]
    }}
    ```
    严禁通过 final_answer 输出除 JSON 字符串之外的任何多余解释性文本！
    """

    print(f"🚀 正在启动 {paper_number} 篇文献深度检索任务，主题: {question} ...\n(此过程可能需要几分钟到十几分钟，请耐心等待)")
    
    # 启动 Agent
    answer = agent.run(task_prompt)

    # 捕获结果并处理 JSON
    try:
        # 去除大模型可能生成的 markdown 代码块标记 (如 ```json)
        cleaned_answer = str(answer).strip()
        if cleaned_answer.startswith("```json"):
            cleaned_answer = cleaned_answer[7:]
        if cleaned_answer.startswith("```"):
            cleaned_answer = cleaned_answer[3:]
        if cleaned_answer.endswith("```"):
            cleaned_answer = cleaned_answer[:-3]
            
        # 尝试解析 JSON
        parsed_data = json.loads(cleaned_answer.strip())
        
        # 写入本地文件
        output_file = f"search_results_{paper_number}_papers.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 成功！{paper_number}篇文献的深度解析 JSON (含关键词) 已保存至 -> {output_file}")
        
    except json.JSONDecodeError as e:
        print(f"\n⚠️ 警告: Agent 返回的结果未能通过 JSON 严格校验。可能由于格式错误。")
        print(f"JSON 解析错误信息: {e}")
        # 即使失败，也将原始内容保存下来，避免搜索了半天的心血白费
        fallback_file = "search_results_raw.txt"
        with open(fallback_file, "w", encoding="utf-8") as f:
            f.write(str(answer))
        print(f"原始内容已备份至 -> {fallback_file}，请手动查看修复。")
def main():
    search("大模型领域前沿进展")


if __name__ == "__main__":
    main()