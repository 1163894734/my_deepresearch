import os
import json
from typing import Dict, List
from smolagents import Tool

class SaveReportTool(Tool):
    name = "save_report"
    description = "用于将生成的 JSON 数据和 Markdown 研报保存到本地。必须传入 approved_clusters, markdown_content 和 run_dir。"
    
    inputs = {
        "clusters_data": {"type": "array", "description": "获批的聚类簇列表数据 (approved_clusters)"},
        "markdown_content": {"type": "string", "description": "生成的最终 Markdown 研报字符串"},
        "run_dir": {"type": "string", "description": "本次运行的输出目录路径"}
    }
    output_type = "string"

    def forward(self, clusters_data: List[Dict], markdown_content: str, run_dir: str) -> str:
        os.makedirs(run_dir, exist_ok=True)
        
        json_path = os.path.join(run_dir, "frontier_identification_report.json")
        md_path = os.path.join(run_dir, "frontier_identification_report.md")

        # 写入 JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"clusters": clusters_data}, f, ensure_ascii=False, indent=2)

        # 写入 Markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        success_msg = f"✅ 数据已成功持久化保存至: {run_dir}"
        print(success_msg)
        return success_msg