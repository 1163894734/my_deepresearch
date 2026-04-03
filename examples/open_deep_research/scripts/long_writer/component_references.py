from __future__ import annotations
import sys
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ..long_writer_agent_v3 import LongWriterAgent

from .base_component import JsonWorkflowComponent
from .cli_debugger import run_component_cli

class ReferencesWritingComponent(JsonWorkflowComponent):
    """参考文献撰写组件：输入引用库，输出格式化后的参考文献排版文本。"""

    name = "references_writing"
    description = "清洗规范化引用数据，并生成最终的参考文献章节文本。"
    input_format = {
        "section": "Dict[str, Any]",
        "available_citations": "Dict[str, Dict[str, Any]]",
        "task": "str",
    }
    output_format = {
        "content": "str",
    }

    @staticmethod
    def _normalize_available_citations(raw: Any) -> Dict[str, Dict[str, Any]]:
        """规范化组件输入中的引用字典/列表结构"""
        if isinstance(raw, dict):
            normalized = {}
            for k, v in raw.items():
                info = dict(v) if isinstance(v, dict) else {"title": str(v)}
                title = str(info.get("title") or k).strip()
                if not title:
                    continue
                info["title"] = title
                authors, year = str(info.get("authors", "")).strip(), str(info.get("year", "")).strip()
                if authors and year:
                    info.setdefault("canonical_key", f"{authors} ({year})")
                normalized[title] = info
            return normalized

        if isinstance(raw, list):
            normalized = {}
            for item in raw:
                if not isinstance(item, dict):
                    continue
                apa, authors, year = str(item.get("apa_citation", "")).strip(), str(item.get("authors", "")).strip(), str(item.get("year", "")).strip()
                title = str(item.get("title", "")).strip() or apa
                if not title:
                    continue
                info = dict(item)
                info["title"] = title
                if authors and year:
                    info.setdefault("canonical_key", f"{authors} ({year})")
                normalized[title] = info
            return normalized

        return {}

    @staticmethod
    def format_citations_to_text(citations_dict: Dict[str, Dict[str, Any]]) -> str:
        """将规范化后的引用字典，按照统一规范格式化为最终文本"""
        if not citations_dict:
            return "暂无引用文献。"
            
        # 按照作者名称首字母和年份排序
        sorted_refs = sorted(
            citations_dict.items(), 
            key=lambda x: (str(x[1].get('authors', '')).lower(), str(x[1].get('year', '')))
        )
        
        # 组装格式化字符串
        formatted_lines = []
        for idx, (_, info) in enumerate(sorted_refs, 1):
            authors = info.get('authors', '未知作者')
            year = info.get('year', 'n.d.')
            title = info.get('title', 'Title unavailable')
            formatted_lines.append(f"[{idx}] {authors}. ({year}). {title}.")
            
        return "\n".join(formatted_lines)


    def run(self, agent: "LongWriterAgent", payload: Dict[str, Any]) -> Dict[str, Any]:
        """核心处理流：输入清洗 -> 格式化组装 -> 返回结果"""
        
        # 1. 安全提取 Payload
        payload_obj = payload.get("input", payload) if isinstance(payload, dict) else {}
        payload_obj = payload_obj if isinstance(payload_obj, dict) else {}

        # 2. 清洗传入的引用数据
        raw_citations = payload_obj.get("available_citations", {})
        normalized_citations = self._normalize_available_citations(raw_citations)

        # 3. 本地直接生成排版好的字符串，不再反向调用 agent
        content = self.format_citations_to_text(normalized_citations)

        return {"content": content}

if __name__ == "__main__":
    sys.exit(run_component_cli(ReferencesWritingComponent()))