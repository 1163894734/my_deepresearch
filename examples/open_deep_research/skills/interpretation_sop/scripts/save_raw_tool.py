import os
from smolagents import Tool

class SaveRawMaterialsTool(Tool):
    name = "save_raw_materials_tool"
    description = "用于将检索到的原始资料（纯文本格式）安全地保存到本地文件中。"
    
    inputs = {
        "content": {"type": "string", "description": "需要保存的文本内容（如 raw_materials）"},
        "filename": {"type": "string", "description": "文件名，例如 'raw_materials.txt'"},
        "run_dir": {"type": "string", "description": "本次运行的输出目录路径"}
    }
    output_type = "string"

    def forward(self, content: str, filename: str, run_dir: str) -> str:
        os.makedirs(run_dir, exist_ok=True)
        file_path = os.path.join(run_dir, filename)

        # 在 Tool 内部使用 open 是绝对安全的，不会被沙盒拦截
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        success_msg = f"✅ 资料已成功保存至: {file_path}"
        print(success_msg)
        return success_msg