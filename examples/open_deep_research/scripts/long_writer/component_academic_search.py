from __future__ import annotations
import sys
from smolagents.monitoring import LogLevel

# ===== 修复导入路径 =====
try:
    from .base_component import JsonWorkflowComponent
    from .cli_debugger import run_component_cli
    from .academic_search_service import AcademicSearchService  # 👈 这是干活的服务类
except ImportError:
    from base_component import JsonWorkflowComponent
    from cli_debugger import run_component_cli
    from examples.open_deep_research.scripts.long_writer.academic_search_service import AcademicSearchService


# 🚨 核心修复：把组件的名字改成 Component，千万不能和服务类同名！
class AcademicSearchComponent(JsonWorkflowComponent):
    """基于双引擎的精准学术检索组件"""

    name = "academic_search"
    description = "使用大模型解析自然语言查询，并通过 API 获取精准的学术文献及引用格式。"
    input_format = {
        "task": "str, 用户的自然语言或逻辑查询词"
    }
    output_format = {
        "task": "str",
        "search_intent": "dict",
        "available_citations": "Dict[str, Dict]"  
    }

    def run(self, agent, payload: dict) -> dict:
        task = str(payload.get("task", "")).strip()
        
        # 顺手把外层的日志也搞成双引擎动态显示
        engine = agent.state.get('search_engine', 'arxiv').upper()

        # 1. 意图解析 (提取查询词和年份) -> 此时调用的就是真正导入的服务类了！
        intent = AcademicSearchService.parse_search_intent(agent.model,task)
        
        search_query = intent.get("search_query", task)
        year_start = intent.get("year_start", 0)
        year_end = intent.get("year_end", 0)
        
        # 2. 调用 API 获取真实论文 (修复了传参方式)
        # 💡 注意：如果你在服务类里把这个方法改名成了 search_academic_papers，这里要跟着改！
        papers_dict = AcademicSearchService.search_academic_papers(
            agent=agent,
            search_query=search_query,
            year_start=str(year_start) if year_start else "",
            year_end=str(year_end) if year_end else "",
            max_fetch=50, 
            target_count=10
        )

        return {
            "task": task,
            "search_intent": intent,
            "available_citations": papers_dict,
            "engine": engine,
            "retrieved_count": len(papers_dict)
        }

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        payload_json_text = r'''{
            "task": "大模型领域，只看最近2年的论文"
        }'''
        # 下面这里的实例化也要改成 Component
        run_component_cli(AcademicSearchComponent(), argv=["--input-json", payload_json_text])
    else:
        run_component_cli(AcademicSearchComponent())