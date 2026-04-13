import json
from scripts.long_writer.section_writing_service import SectionWritingService
from scripts.citation_validator import CitationValidator

class SectionWriterAgent:
    def __init__(self, context):
        self.ctx = context
        self.validator = CitationValidator(model=self.ctx.model, remove_entire_invalid_sentence=False)
        
    def write_section(self, task: str, global_outline: str, section: dict, base_citations: dict) -> dict:
        section_title = section.get('title', '未命名章节')
        self.ctx.logger.info(f"-> [Writer] 开始处理章节: {section_title}")
        
        # 1. 细粒度检索
        academic_comp = self.ctx.components.get("academic_search")
        if academic_comp:
            fine_rag_res = academic_comp.run(self.ctx, {"task": f"{section_title} {section.get('goal', '')}"})
            fine_rag_context = json.dumps(fine_rag_res.get("available_citations", {}), ensure_ascii=False)
        else:
            fine_rag_context = "{}"
            
        payload = {
            "section": section,
            "available_citations": base_citations.copy(),
            "fine_rag_context": fine_rag_context,
            "task": task,
            "outline": global_outline,
            "prev_section_content": "" 
        }
        
        # 🔥 修复核心：这里必须传入 self.ctx.tools 而不是 components
        skeleton = SectionWritingService.plan_body_skeleton(self.ctx, payload)
        payload["skeleton"] = skeleton
        
        # 🔥 修复核心：这里也必须传入 self.ctx.tools
        raw_content = SectionWritingService.compose_body_content(self.ctx, payload)
        
        validated_citations, final_content, step_logs = self.validator.run_five_step_validation(
            base_citations.copy(), raw_content, section_title
        )
        if hasattr(self.ctx.workspace, 'append_validation_logs'):
            self.ctx.workspace.append_validation_logs(step_logs)
        
        # 👇 ========== 新增的终端打印逻辑 ========== 👇
        if step_logs:
            self.ctx.logger.info(f"🛡️ [引用校验] 章节《{section_title}》五步验证完毕:")
            for log in step_logs:
                step = log.get('step', '?')
                action = str(log.get('action', 'unknown')).ljust(8)
                cost = log.get('elapsed_ms', 0)
                out_sum = log.get('output_summary', {})
                total = out_sum.get('citation_total', 0)
                verified = out_sum.get('verified_count', 0)
                supported = out_sum.get('claim_supported_count', 0)
                
                self.ctx.logger.info(
                    f"    ├─ 步骤 {step} [{action}] 耗时 {cost}ms | "
                    f"剩余文献: {total}篇 (在线核实:{verified}, 观点支持:{supported})"
                )
        # 👆 ========== 新增的终端打印逻辑 ========== 👆

        self.ctx.logger.info(f"✅ [Writer] 章节完成: {section_title} ({len(final_content)} 字)")
        self.ctx.workspace.append_section(section_title, final_content)
        for citation_key, citation_data in validated_citations.items():
            trace_details = citation_data.get("validation_result", {}).get("trace_details", [])
            for trace in trace_details:
                if trace["is_supported"]:
                    self.ctx.logger.info(f"    🔗 [溯源成功] 句子: {trace['sentence'][:20]}...")
                    self.ctx.logger.info(f"       文献: {citation_key}")
                    self.ctx.logger.info(f"       原文: {trace['evidence_text'][:50]}...")
        return {
            "section": section,
            "content": final_content,
            "new_citations": validated_citations
        }