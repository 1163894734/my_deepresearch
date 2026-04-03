import argparse
import json
import os
import threading
from dotenv import load_dotenv

# 核心模型与工具
from smolagents import OpenAIModel, DuckDuckGoSearchTool
from scripts.skill_loader import load_skills_from_directory
from scripts.identification_agent import IdentificationAgent
from scripts.test_identification_agent_simple import get_highly_realistic_docs

payload = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts/test_payload.json"), "r", encoding="utf-8"))
# 由于不再使用 Manager 去分发任务，修改 Prompt 直接让该 Agent 执行数据处理
test_prompt = f"""
请帮我执行一项【前沿技术识别】任务。
请读取以下 JSON 格式的输入数据，进行筛选、聚类识别、专家投票等环节。
完成后，请为我输出一份通俗易懂的前沿技术识别报告。

【输入数据 Payload】
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

load_dotenv(override=True)

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

# 保留原有的 Browser Config 和 downloads 文件夹创建逻辑，以防 custom_tools 中有依赖
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question", type=str, nargs="?", default=test_prompt, help="Optional specific question",
    )
    return parser.parse_args()


def create_identification_agent():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 保持原有的模型配置不变
    model = OpenAIModel(
        model_id="Qwen3-235B-A22B-Instruct-2507",
        api_base="https://llmapi.paratera.com/v1",
        api_key=os.environ["DYM_API_KEY"],
        max_tokens=8192
    )
    small_model = OpenAIModel(
        model_id="local-model",  # 这里的 ID 可以是任意字符串，LM Studio 会使用你当前加载的模型
        api_base="http://localhost:1234/v1",  # LM Studio 的默认 API 地址
        api_key="not-needed",  # 本地运行通常不需要 key
    )

    
    # 保持技能加载逻辑不变
    skills_dir = os.path.join(base_dir, "skills")
    print(f"Loading skills from {skills_dir}...")
    custom_tools = load_skills_from_directory(skills_dir, model=model)
    print(f"Loaded {len(custom_tools)} custom tools: {[t.name for t in custom_tools]}")

    # 基础网络搜索工具
    web_search_tool = DuckDuckGoSearchTool()
    web_search_tool.name = "web_search"
    
    # 直接创建并返回 IdentificationAgent
    identification_agent = IdentificationAgent(
        model=model,
        small_model = small_model,
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

    return identification_agent


def main():
    args = parse_args()

    # 初始化单个 Agent
    agent = create_identification_agent()


    
    print("🚀 [系统启动] 正在直接调用 Identification Agent...")
    answer = agent.run(test_prompt)
    print(f"Got this answer:\n{answer}")


if __name__ == "__main__":
    main()