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

    def forward(self, query: str, engine: str = "arxiv") -> str:
            # 1. 临时设置 agent 状态（模拟 service 需要的环境）
            class MockAgent:
                def __init__(self, engine_name):
                    self.state = {"search_engine": engine_name or "arxiv", "search_sort": "date"}
                    
                    # 定义一个简单的模拟 Logger 对象
                    class SimpleLogger:
                        def log(self, message, level=LogLevel.INFO):
                            # 简单的将日志打印到控制台
                            print(f"[{level}] {message}")
                    
                    self.logger = SimpleLogger()

            mock_agent = MockAgent(engine or "arxiv")
            
            # 2. 调用核心服务逻辑
            papers = AcademicSearchService.search_academic_papers(
                mock_agent, 
                search_query=query,
                target_count=5
            )
            
            if not papers:
                return "未找到相关文献。"
                
            # 3. 统一返回格式化后的字符串，方便 Agent 阅读
            results = []
            for title, info in papers.items():
                results.append(f"标题: {title}\n作者: {info['authors']}\n年份: {info['year']}\n摘要: {info['abstract']}\n引用格式: {info['apa_citation']}\n")
            
            return "\n---\n".join(results)