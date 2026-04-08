import json
import re
from concurrent.futures import ThreadPoolExecutor
from smolagents import Tool

class JudgeAllDraftsTool(Tool):
    name = "judge_all_drafts_tool"
    description = "并发对初稿字典中的所有版本进行事实核查与审修。返回定稿字典。"
    inputs = {
        "outline": {"type": "string", "description": "评估大纲"},
        "facts": {"type": "string", "description": "客观事实底稿"},
        "drafts_dict": {"type": "any", "description": "包含三版初稿的字典"}
    }
    output_type = "any"

    def __init__(self, model=None, **kwargs):
        super().__init__()
        self.model = model

    def forward(self, outline: str, facts: str, drafts_dict: dict) -> dict:
        if not self.model:
            raise ValueError("JudgeAllDraftsTool 需要被注入 model。")

        def judge_single(style_name: str, draft_text: str) -> tuple:
            judge_prompt = f"""
            你是一名严厉的编辑。请根据以下大纲和事实检查稿件。
            【大纲】: {outline}
            【事实】: {facts}
            【稿件】: {draft_text}

            检查标准：1. 事实是否准确 2. 是否完全覆盖大纲维度。
            请直接输出 JSON，格式：{{"pass": true/false, "revision_instruction": "修改建议"}}
            """
            
            judge_resp = self.model([{"role": "user", "content": [{"type": "text", "text": judge_prompt}]}])
            raw_judge = judge_resp.content if hasattr(judge_resp, "content") else str(judge_resp)
            
            try:
                m = re.search(r"\{.*\}", raw_judge, re.DOTALL)
                data = json.loads(m.group(0)) if m else {"pass": True}
                
                if data.get("pass") is True:
                    return style_name, draft_text
                    
                instruction = data.get("revision_instruction", "对齐大纲进行优化。")
                revise_prompt = f"请根据建议优化稿件，直接输出正文。\n建议：{instruction}\n原稿：{draft_text}"
                revise_resp = self.model([{"role": "user", "content": [{"type": "text", "text": revise_prompt}]}])
                final_text = revise_resp.content if hasattr(revise_resp, "content") else str(revise_resp)
                return style_name, final_text
                
            except Exception:
                return style_name, draft_text # 兜底

        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(judge_single, name, text) for name, text in drafts_dict.items()]
            for future in futures:
                name, content = future.result()
                results[name] = content
                
        return results