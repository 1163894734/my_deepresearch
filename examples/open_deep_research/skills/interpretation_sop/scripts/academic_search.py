import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET
import datetime
import time
from typing import Dict, Any
from smolagents import Tool

class AcademicSearchTool(Tool):
    name = "academic_search_tool"
    description = "真实的学术文献检索工具，支持检索最新论文。传入关键词获取文献摘要集合。"
    inputs = {
        "query": {
            "type": "string", 
            "description": "英文检索关键词（建议提炼后的核心词）"
        },
        "engine": {
            "type": "string", 
            "description": "搜索引擎，可选 'openalex' 或 'arxiv'，默认 'openalex'",
            "nullable": True  # <--- 加上这一行来解决报错
        }
    }
    output_type = "string"

    def __init__(self, model=None, **kwargs):
        super().__init__()

    def forward(self, query: str, engine: str = "openalex") -> str:
        current_year = datetime.datetime.now().year
        y_start, y_end = current_year - 3, current_year
        max_fetch, target_count = 10, 3
        citations_dict: Dict[str, Dict[str, Any]] = {}

        req = None
        # 如果大模型没有传 engine 参数，它可能是 None，这里做个兜底
        current_engine = engine if engine else "openalex"

        if current_engine.lower() == "openalex":
            encoded_query = urllib.parse.quote(query)
            url = f"https://api.openalex.org/works?search={encoded_query}&filter=publication_year:{y_start}-{y_end}&sort=publication_date:desc&per-page={max_fetch}"
            headers = {'User-Agent': 'mailto:open_deep_research@example.com'}
            req = urllib.request.Request(url, headers=headers)
        else:
            encoded_query = urllib.parse.quote(f'all:{query}')
            url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_fetch}&sortBy=submittedDate&sortOrder=descending"
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    if current_engine.lower() == "openalex":
                        data = json.loads(response.read().decode('utf-8'))
                        for paper in data.get('results', []):
                            if len(citations_dict) >= target_count: break
                            title = str(paper.get('title') or "").strip()
                            if title:
                                citations_dict[title] = {"title": title, "year": str(paper.get('publication_year') or y_end)}
                    else:
                        root = ET.fromstring(response.read().decode('utf-8'))
                        ns = {'atom': 'http://www.w3.org/2005/Atom'}
                        for entry in root.findall('atom:entry', ns):
                            if len(citations_dict) >= target_count: break
                            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                            citations_dict[title] = {"title": title, "year": str(y_end)}
                break 
            except Exception:
                if attempt < max_retries - 1: time.sleep(2)

        if not citations_dict:
            return "未检索到相关的学术文献。"
        
        result_text = f"【{current_engine.upper()} 学术检索结果】\n"
        for i, info in enumerate(citations_dict.values()):
            result_text += f"[{i+1}] 标题：{info['title']} ({info['year']})\n"
        return result_text