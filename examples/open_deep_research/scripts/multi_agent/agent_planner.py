from scripts.long_writer.outline_parsing_service import OutlineParsingService

class PlannerAgent:
    def __init__(self, context):
        self.ctx = context  # 注入工具箱
        self.outline_parser = OutlineParsingService()
        
    def generate_plan(self, task: str) -> dict:
        self.ctx.logger.info("🗺️ [Planner] 开始执行粗粒度检索...")
        
        search_comp = self.ctx.components.get("keyword_search_expansion")
        
        # 🔥 灵魂操作：调用老代码时，传入 ctx 代替 self！老代码以为这就是个 Agent！
        search_result = search_comp.run(self.ctx, {"task": task}) if search_comp else {"available_citations": {}}
        citations = search_result.get("available_citations", {})
        
        self.ctx.logger.info("🗺️ [Planner] 开始生成结构大纲...")
        outline_comp = self.ctx.components.get("outline_generation")
        outline_res = outline_comp.run(self.ctx, {"task": task, "available_citations": citations})
        
        final_outline = outline_res.get("outline_v1", "")
        sections = self.outline_parser.parse_sections(final_outline, self.ctx.logger.info)
        
        return {
            "outline": final_outline,
            "citations": citations,
            "sections": sections
        }