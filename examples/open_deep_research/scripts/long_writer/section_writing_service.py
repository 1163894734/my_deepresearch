# 文件路径: open_deep_research/scripts/long_writer/section_writing_service.py

import argparse
import json
import re
import os
from typing import Any, Dict, Optional, TYPE_CHECKING

# 🚀 深度结合你的 common_utils
from utils.common_utils import ModelProvider, execute_tool_call

if TYPE_CHECKING:
    from scripts.multi_agent.agent_context import PipelineContext

class SectionWritingService:

    @staticmethod
    def _json_dumps(data: Any) -> str:
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _build_canonical_citation_key(authors: str, year: str, fallback_title: str = "") -> str:
        authors_text = str(authors or "").strip()
        year_text = str(year or "").strip()
        if authors_text and year_text:
            return f"{authors_text} ({year_text})"
        return str(fallback_title or "").strip()

    @staticmethod
    def _trim_text(text: str, max_chars: int) -> str:
        if not text: 
            return ""
        return text if len(text) <= max_chars else text[:max_chars] + "\n...(已截断)"

    # =========================================================================
    # 🚀 新增：基于 RAG 和本地大模型的终极章节撰写器
    # =========================================================================
    @staticmethod
    def generate_section_with_rag(
        chapter_node: dict, 
        available_tools: dict, 
        sop_path: str = "skills/long_writer_section_sop/SKILL.md",
        logger=None
    ) -> str:
        """
        深度结合 RAG 工具和本地大模型，按图施工撰写单个章节。
        """
        # 1. 解析大纲节点信息
        title = chapter_node.get("chapter_title") or chapter_node.get("section_title") or chapter_node.get("subsection_title", "未知标题")
        argument = chapter_node.get("core_argument", "")
        papers = chapter_node.get("supporting_papers", [])
        
        # 区分本地成功文献和仅在线文献
        local_papers = [{"id": p["id"], "local_path": p["local_path"]} for p in papers if p.get("status") == "local_success" and p.get("local_path")]
        online_urls = [p["url"] for p in papers if p.get("status") == "online_only"]
        
        # 2. 调度 fine_rag 提取高浓度语料
        rag_context = ""
        if local_papers:
            if logger: logger.info(f"[{title}] 启动 RAG 检索，扫描 {len(local_papers)} 篇本地文献...")
            try:
                # 完美调用你的 execute_tool_call
                rag_context = execute_tool_call(
                    tool_name="fine_rag", 
                    arguments={"query": argument, "papers": local_papers, "top_k": 6}, 
                    available_tools=available_tools, 
                    logger=logger
                )
            except Exception as e:
                if logger: logger.error(f"[{title}] RAG 检索失败: {e}")
                rag_context = f"RAG 提取失败: {str(e)}"
        else:
            rag_context = "无本地可用的 PDF 语料，请依据内部学术知识及提供的在线URL进行学术推演。"
            
        # 3. 读取严格的 SOP 提示词
        try:
            with open(sop_path, "r", encoding="utf-8") as f:
                sys_prompt = f.read()
        except Exception:
            sys_prompt = "你是一个顶尖的学术专家，请严格基于给定的 RAG 语料撰写高学术密度的段落，并在句末使用 [id] 进行引用。"
            
        # 4. 构建 Prompt
        user_prompt = (
            f"【待撰写章节】: {title}\n"
            f"【核心论点】: {argument}\n\n"
            f"【RAG 精准提取语料】:\n{rag_context}\n\n"
            f"【在线扩展链接】:\n{online_urls}\n\n"
            f"请直接输出正文，不要输出章节标题。"
        )
        
        if logger: logger.info(f"[{title}] 语料组装完毕，正在调用 ModelProvider 大模型...")
        
        # 5. 调用你的统一大模型接口
        model = ModelProvider.get_model("main")
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = model(messages)
            # 兼容 smolagents Message 的返回结构
            content = getattr(response, 'content', str(response)) 
            if logger: logger.info(f"[{title}] ✅ 章节撰写完成！")
            return content
        except Exception as e:
            if logger: logger.error(f"[{title}] ❌ 大模型生成失败: {e}")
            return f"> 撰写失败: {str(e)}"

    @staticmethod
    def write_intro(context: "PipelineContext", payload: dict) -> str:
        """[原子服务] 仅负责调用引言撰写技能并返回文本"""
        return str(context.execute_tool_call(
            "introduction_write",
            {"input": SectionWritingService._json_dumps(payload)}
        ))

    @staticmethod
    def write_conclusion(context: "PipelineContext", payload: dict) -> str:
        """[原子服务] 仅负责调用结论撰写技能并返回文本"""
        if "main_text" not in payload and "full_text" in payload:
            payload["main_text"] = payload["full_text"]
            
        return str(context.execute_tool_call(
            "conclusion_write",
            {"input": SectionWritingService._json_dumps(payload)}
        ))

    @staticmethod
    def write_abstract(context: "PipelineContext", payload: dict) -> str:
        """[原子服务] 仅负责调用摘要撰写技能并返回文本"""
        if "conclusion_text" not in payload:
            payload["conclusion_text"] = payload.get("conclusion_text") or payload.get("full_text") or payload.get("main_text") or ""
            
        return str(context.execute_tool_call(
            "abstract_write",
            {"input": SectionWritingService._json_dumps(payload)}
        ))

    @staticmethod
    def plan_body_skeleton(context: "PipelineContext", payload: dict) -> str:
        """[原子服务] 仅负责调用正文骨架规划技能并返回骨架"""
        return str(context.execute_tool_call(
            "section_skeleton_planning",
            {"input": SectionWritingService._json_dumps(payload)}
        ))

    @staticmethod
    def compose_body_content(context: "PipelineContext", payload: dict) -> str:
        """[原子服务] 仅负责基于骨架调用文本组装技能并返回正文"""
        return str(context.execute_tool_call(
            "section_composition_styling",
            {"input": SectionWritingService._json_dumps(payload)}
        ))

    @staticmethod
    def run_fine_rag_web_search(
        context: "PipelineContext",
        search_query: str, 
        search_tool_name: str = "web_search",
        max_chars: int = 1600
    ) -> str:
        """[原子服务] 仅负责执行细粒度网页检索解析并返回摘要"""
        result = context.execute_tool_call(
            search_tool_name,
            {"query": search_query}
        )
        if isinstance(result, str) and result.strip():
            raw_result = result.strip()
            markers = [
                "### 1. Task outcome (short version):",
                '"### 1. Task outcome (short version)":',
                '"final_answer"',
                "tool_code",
            ]
            if any(m in raw_result for m in markers):
                short_match = re.search(
                    r"###\s*1\.\s*Task outcome \(short version\):\s*(.*?)(?:\n###\s*2\.|$)",
                    raw_result,
                    re.IGNORECASE | re.DOTALL,
                )
                if short_match:
                    result = short_match.group(1).strip()
                else:
                    json_match = re.search(
                        r'"###\s*1\.\s*Task outcome \(short version\)"\s*:\s*"(.*?)"\s*,\s*"###\s*2\.',
                        raw_result,
                        re.IGNORECASE | re.DOTALL,
                    )
                    result = json_match.group(1).replace("\\n", "\n").strip() if json_match else raw_result
            else:
                result = raw_result

            return SectionWritingService._trim_text(result, max_chars)
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="SectionWritingService 调试入口")
    parser.add_argument("--list-methods", action="store_true", help="列出可调用静态方法")
    args = parser.parse_args()

    if args.list_methods:
        print("SectionWritingService is now a pure atomic tool caller without side-effects.")
        print("\nAvailable public methods:")
        methods = [func for func in dir(SectionWritingService) if callable(getattr(SectionWritingService, func)) and not func.startswith("_")]
        for m in methods:
            print(f" - {m}")
        return 0

    print("这是 service 文件，不直接执行。请在外部主控逻辑（如 Agent 或脚本）中，通过传入 context 对象来调用其静态方法。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())