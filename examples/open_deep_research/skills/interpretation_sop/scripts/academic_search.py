import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET
import datetime
import time
from typing import Dict, Any
from scripts.long_writer.academic_search_service import AcademicSearchService
from smolagents import Tool
from smolagents.monitoring import LogLevel

class AcademicSearchTool(Tool):
    name = "academic_search_tool"
    description = "真实的学术文献检索工具，支持获取领域内高引用经典论文。传入关键词获取文献摘要集合。"
    inputs = {
        "query": {
            "type": "string", 
            "description": "英文检索关键词（建议提炼后的核心词）"
        },
        "engine": {
            "type": "string", 
            "description": "搜索引擎，可选 'openalex' 或 'arxiv'，默认 'openalex'",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, model=None, **kwargs):
        super().__init__()

    # 🔥 修改 1：默认引擎改为 openalex，因为只有 OpenAlex 支持引用量排序
    def forward(self, query: str, engine: str = "openalex") -> str:
            # 1. 临时设置 agent 状态（模拟 service 需要的环境）
            class MockAgent:
                def __init__(self, engine_name):
                    # 🔥 修改 2：强制将排序策略改为 "citation" (按引用量降序)
                    self.state = {"search_engine": engine_name or "openalex", "search_sort": "citation"}
                    
                    # 定义一个简单的模拟 Logger 对象
                    class SimpleLogger:
                        def log(self, message, level=LogLevel.INFO):
                            print(f"[{level}] {message}")
                    
                    self.logger = SimpleLogger()

            mock_agent = MockAgent(engine or "openalex")
            
            # 2. 调用核心服务逻辑
            papers = AcademicSearchService.search_academic_papers(
                mock_agent, 
                search_query=query,
                year_start="1950",  # 🔥 修改 3：传入一个极早的年份 (如 1950)，打破底层默认的 2017 年限制
                year_end=str(datetime.datetime.now().year),
                target_count=5
            )
            
            if not papers:
                return "未找到相关文献。"
                
            # 3. 统一返回格式化后的字符串，方便 Agent 阅读
            results = []
            for title, info in papers.items():
                # 🔥 修改 4：在返回给大模型的文本中，加上“引用量”这一项，让大模型知道这篇论文的权威性
                results.append(f"标题: {title}\n作者: {info['authors']}\n年份: {info['year']}\n被引次数: {info.get('citation_count', 0)}\n摘要: {info['abstract']}\n引用格式: {info['apa_citation']}\n")
            
            return "\n---\n".join(results)