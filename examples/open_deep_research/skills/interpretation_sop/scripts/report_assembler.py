import os
import re
from smolagents import Tool

class AssembleAndSaveReportTool(Tool):
    name = "assemble_and_save_report_tool"
    description = "将生成的各项内容组装成最终的 Markdown 解读报告，并保存到指定的运行目录。"
    inputs = {
        "target_term": {"type": "string", "description": "核心名词"},
        "outline": {"type": "string", "description": "动态评估大纲"},
        "facts": {"type": "string", "description": "客观事实底稿"},
        "judged_drafts": {"type": "any", "description": "包含三版定稿的字典"},
        "run_dir": {"type": "string", "description": "报告保存的目标文件夹路径"}
    }
    output_type = "string"

    def __init__(self, model=None, **kwargs):
        super().__init__()

    def forward(self, target_term: str, outline: str, facts: str, judged_drafts: dict, run_dir: str) -> str:
        clean_drafts = {}
        for k, v in judged_drafts.items():
            clean_text = re.sub(r"^(#+ *(百科版|专报版|科普版|解读报告)).*\n?", "", str(v).strip(), flags=re.IGNORECASE).strip()
            clean_drafts[k] = clean_text or str(v)

        parts = [
            f"# 技术名词深度解读：{target_term}", 
            "\n## 1. 动态评估大纲\n", outline,
            "\n## 2. 客观事实底稿\n", facts,
            "\n## 3. 多维深度解读",
            "\n### [百科版]\n", clean_drafts.get("百科版", "（缺失）"),
            "\n### [专报版]\n", clean_drafts.get("专报版", "（缺失）"),
            "\n### [科普版]\n", clean_drafts.get("科普版", "（缺失）"),
        ]
        
        output_markdown = "\n".join(parts)
        
        # 确保目录存在
        os.makedirs(run_dir, exist_ok=True)
        
        # 替换非法字符生成文件名
        safe_term = "".join([c for c in target_term if c.isalnum() or c in [' ', '_']]).strip().replace(' ', '_')
        output_file = os.path.join(run_dir, f"interpretation_report_{safe_term}.md")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_markdown)
            
        return f"✅ 报告组装完成并成功保存至: {output_file}"