from scripts.long_writer.section_writing_service import SectionWritingService

class FinalizerAgent:
    def __init__(self, context):
        self.ctx = context
        
    def compile_report(self, task: str, global_outline: str, compiled_sections: list, all_citations: dict) -> str:
        full_text_body = "\n\n".join([s["content"] for s in compiled_sections])
        
        self.ctx.logger.info("-> [Finalizer] 正在生成全局摘要...")
        
        # 🔥 修复核心：这里必须传入 self.ctx.tools
        abstract_content = SectionWritingService.write_abstract(
            context=self.ctx,
            payload={"full_text": full_text_body, "task": task}
        )
        
        references_content = self._format_references(all_citations)
        
        final_md = f"# {task}\n\n## 📋 文章大纲\n{global_outline}\n\n---\n\n"
        final_md += f"## 摘要\n\n{abstract_content}\n\n"
        for sec in compiled_sections:
            final_md += f"## {sec['section'].get('title', '未知章节')}\n\n{sec['content']}\n\n"
        final_md += f"## 参考文献\n\n{references_content}\n\n"
        
        return final_md

    def _format_references(self, citations: dict) -> str:
        if not citations: return "暂无引用文献。"
        sorted_refs = sorted(citations.items(), key=lambda x: (str(x[1].get('authors', '')).lower(), str(x[1].get('year', ''))))
        return "\n".join([f"[{idx}] {info.get('authors', '未知')}. ({info.get('year', 'n.d.')}). {info.get('title', '')}." for idx, (_, info) in enumerate(sorted_refs, 1)])