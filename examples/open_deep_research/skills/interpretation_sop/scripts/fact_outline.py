from smolagents import Tool

class ExtractFactAndOutlineTool(Tool):
    name = "extract_fact_and_outline_tool"
    description = "根据检索到的原始资料，提炼出客观事实底稿并生成动态评估大纲。返回包含 facts 和 outline 的字典。"
    inputs = {
        "target_term": {
            "type": "string", 
            "description": "需要解读的核心名词"
        },
        "raw_materials": {
            "type": "string", 
            "description": "检索工具返回的所有原始文本资料"
        }
    }
    output_type = "any" # 允许返回 dict

    def __init__(self, model=None, **kwargs):
        super().__init__()
        self.model = model # 接收由 skill_loader 注入的大模型实例

    def forward(self, target_term: str, raw_materials: str) -> dict:
        if not self.model:
            raise ValueError("ExtractFactAndOutlineTool 需要被注入 model 才能运行。")
            
        # 提炼事实
        prompt_facts = f"归纳事实底稿（客观、无评价）。名词：{target_term}\n资料：{raw_materials[:6000]}"
        facts_resp = self.model([{"role": "user", "content": [{"type": "text", "text": prompt_facts}]}])
        facts = facts_resp.content if hasattr(facts_resp, "content") else str(facts_resp)
        
        # 生成大纲
        prompt_outline = f"为【{target_term}】生成评估大纲（3-5个维度）。\n事实：{facts}"
        outline_resp = self.model([{"role": "user", "content": [{"type": "text", "text": prompt_outline}]}])
        outline = outline_resp.content if hasattr(outline_resp, "content") else str(outline_resp)
        
        return {"facts": facts, "outline": outline}