import argparse
import os
import json
import re
import datetime
import time
import socket
import urllib.request
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, Any
from dotenv import load_dotenv

# 核心模型与工具
from utils import common_utils
from smolagents import OpenAIModel, DuckDuckGoSearchTool, Tool, ChatMessage, MessageRole

# 导入咱们改好的 InterpretationAgent
from scripts.interpretation_agent import InterpretationAgent

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


# =====================================================================
# 真实的学术检索工具 (完美融入你提供的 AcademicSearchService 逻辑)
# =====================================================================
class AcademicSearchTool(Tool):
    name = "coarse_rag"
    description = "真实的学术文献检索工具，自动将中文意图转为英文关键词，支持检索 ArXiv 最新论文。"
    inputs = {"query": {"type": "string", "description": "用户的原始查询需求"}}
    output_type = "string"

    def __init__(self, model, engine="arxiv"):
        super().__init__()
        self.model = model # 使用小模型进行意图解析
        self.engine = engine.lower()

    @staticmethod
    def _reconstruct_openalex_abstract(inverted_index: dict) -> str:
        """还原 OpenAlex 的倒排索引摘要"""
        if not inverted_index: return ""
        words = []
        for word, positions in inverted_index.items():
            for pos in positions:
                words.append((pos, word))
        words.sort(key=lambda x: x[0])
        return " ".join([w[1] for w in words])

    def forward(self, query: str) -> str:
        current_year = datetime.datetime.now().year
        
        # 1. 意图解析：调用小模型将输入转化为严谨的英文检索词
        prompt = f"""
        请分析用户的检索需求，提取用于学术数据库的检索参数。
        用户输入: "{query}"
        
        要求：
        1. 必须将中文意图翻译为最准确的【英文学术关键词】。
        2. search_query: 提取核心英文搜索词。绝对禁止保留“前沿”、“进展”等虚词！
           所有的搜索词必须拼成【一整个普通字符串】，绝对禁止在内部使用双引号！
        3. year_start / year_end: 提取时间限制。如果要求最新，设为 {current_year - 2} 到 {current_year}。
        
        必须输出合法 JSON：
        {{
            "search_query": "large language model reasoning",
            "year_start": "{current_year - 2}",
            "year_end": "{current_year}"
        }}
        """
        try:
            messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}])]
            response = self.model(messages).content
            m = re.search(r"\{.*\}", str(response), re.DOTALL)
            parsed = json.loads(m.group(0)) if m else {}
            search_query = parsed.get("search_query", query).replace('"', '').strip()
            y_start = int(parsed.get("year_start", 2017))
            y_end = int(parsed.get("year_end", current_year))
            print(f"🔍 [学术引擎] 意图解析成功 -> 关键词: '{search_query}', 年份: {y_start}-{y_end}")
        except Exception as e:
            print(f"⚠️ [学术引擎] 意图解析失败，使用默认参数: {e}")
            search_query = query.replace('"', '').strip()
            y_start = 2017
            y_end = current_year

        if not search_query:
            return "无效的查询词。"

        citations_dict: Dict[str, Dict[str, Any]] = {}
        max_fetch = 15
        target_count = 5

        # 2. 组装请求对象
        req = None
        if self.engine == "openalex":
            encoded_query = urllib.parse.quote(search_query)
            url = f"https://api.openalex.org/works?search={encoded_query}&filter=publication_year:{y_start}-{y_end}&sort=publication_date:desc&per-page={max_fetch}"
            headers = {'User-Agent': 'mailto:open_deep_research@example.com'}
            req = urllib.request.Request(url, headers=headers)
        else:
            encoded_query = urllib.parse.quote(f'all:{search_query}')
            url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_fetch}&sortBy=submittedDate&sortOrder=descending"
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)

        # 3. 三次抗抖动重试检索
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    if self.engine == "openalex":
                        data = json.loads(response.read().decode('utf-8'))
                        for paper in data.get('results', []):
                            if len(citations_dict) >= target_count: break
                            title = str(paper.get('title') or "").strip()
                            abstract = self._reconstruct_openalex_abstract(paper.get('abstract_inverted_index', {}))
                            if title and abstract:
                                citations_dict[title] = {
                                    "title": title, "abstract": abstract, 
                                    "year": str(paper.get('publication_year') or y_end),
                                    "url": paper.get('id', '')
                                }
                    else:
                        root = ET.fromstring(response.read().decode('utf-8'))
                        ns = {'atom': 'http://www.w3.org/2005/Atom'}
                        for entry in root.findall('atom:entry', ns):
                            if len(citations_dict) >= target_count: break
                            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                            abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                            url_str = entry.find('atom:id', ns).text.strip()
                            published = entry.find('atom:published', ns).text
                            year = int(published.split('-')[0]) if published else y_end
                            if y_start <= year <= y_end:
                                citations_dict[title] = {
                                    "title": title, "abstract": abstract, 
                                    "year": str(year), "url": url_str
                                }
                print(f"✅ [学术引擎] {self.engine.upper()} 检索成功！获取到 {len(citations_dict)} 篇有效文献。")
                break 
            except Exception as e:
                print(f"⚠️ [学术引擎] 网络波动或解析异常 ({attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1: time.sleep(2)

        # 4. 格式化输出为纯文本供 Agent 使用
        if not citations_dict:
            return "未检索到相关的学术文献，请尝试更换关键词。"
        
        result_text = f"【{self.engine.upper()} 学术文献库检索结果】\n"
        for i, info in enumerate(citations_dict.values()):
            # 截断超长摘要，防止冲爆 Token
            result_text += f"[{i+1}] 标题：{info['title']} ({info['year']})\n摘要：{info['abstract'][:600]}...\n链接：{info['url']}\n\n"
        
        return result_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question", type=str, nargs="?", default="具身智能", help="需要解读的技术名词"
    )
    return parser.parse_args()


def create_interpretation_agent():
    # 1. 大模型 (主笔/裁判)
    model = common_utils.ModelProvider.get_model()
    
    # 2. 小模型 (意图解析/结构化抽取)
    small_model = common_utils.ModelProvider.get_model("small")

    # 3. 基础网络搜索工具 + 真实的学术检索工具
    web_search_tool = DuckDuckGoSearchTool()
    web_search_tool.name = "web_search"
    
    # 将小模型传给学术工具，用于提炼英文关键词。默认使用 arxiv 引擎，也可改为 openalex
    academic_tool = AcademicSearchTool(model=model, engine="openalex")
    
    tools = [academic_tool, web_search_tool]

    # 4. 创建并返回 InterpretationAgent
    interpretation_agent = InterpretationAgent(
        model=model,
        small_model=small_model,
        tools=tools,
        max_steps=20,
        verbosity_level=2,
        name="interpretation_agent",
        description="""进行技术名词深度解读时使用。
适用场景：
- 需要并行检索本地学术库和互联网时效信息
- 需要输出客观事实摘要
- 需要输出百科版、专报版、科普版三种不同风格的解读报告
""",
        provide_run_summary=True,
    )

    return interpretation_agent


def main():
    args = parse_args()
    agent = create_interpretation_agent()

    test_prompt = f"""
请帮我执行一项【技术名词解读】任务。
你需要针对以下概念进行检索、抽取客观事实，并基于动态大纲输出三风格报告。

【目标技术名词】
{args.question}
"""
    
    print(f"🚀 [系统启动] 正在直接调用 Interpretation Agent 解读: {args.question}...")
    answer = agent.run(test_prompt)
    print("\n================ 最终报告 ================\n")
    print(answer)


if __name__ == "__main__":
    main()