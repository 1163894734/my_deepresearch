from concurrent.futures import ThreadPoolExecutor
from smolagents import Tool

class GenerateAllDraftsTool(Tool):
    name = "generate_all_drafts_tool"
    description = "一次性底层并发生成百科版、专报版、科普版三个风格的解读底稿。返回字典格式。"
    inputs = {
        "target_term": {"type": "string", "description": "核心名词"},
        "facts": {"type": "string", "description": "客观事实底稿"},
        "outline": {"type": "string", "description": "动态评估大纲"}
    }
    output_type = "any"

    def __init__(self, model=None, **kwargs):
        super().__init__()
        self.model = model

    def forward(self, target_term: str, facts: str, outline: str) -> dict:
        if not self.model:
            raise ValueError("GenerateAllDraftsTool 需要被注入 model。")

        styles = {
            "百科版": "角色：辞海编辑。风格：学术、严谨、客观。结构：定义、起源、原理、应用、结论。",
            "专报版": "角色：智库分析师。风格：战略、前瞻、精炼。重点：行业趋势、挑战、对策建议。",
            "科普版": "角色：科普作家。风格：生动、通俗、形象。要求：多用比喻，严禁公式。",
        }
        
        def write_single_draft(style_name: str, style_prompt: str) -> tuple:
            prompt = f"""
            {style_prompt}
            请基于以下【客观事实】撰写【{target_term}】的解读稿。
            要求：1. 必须严格遵循【参考大纲】的结构。2. 长度800字左右。直接输出正文。
            【参考大纲】：\n{outline}\n
            【客观事实】：\n{facts}
            """
            resp = self.model([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
            content = resp.content if hasattr(resp, "content") else str(resp)
            return style_name, content

        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(write_single_draft, name, prompt) for name, prompt in styles.items()]
            for future in futures:
                name, content = future.result()
                results[name] = content
                
        return results