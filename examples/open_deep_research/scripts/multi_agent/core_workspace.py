import os
import json
import time

class ReportWorkspace:
    """彻底剥离了 Agent 逻辑，只负责单纯的文件读写"""
    def __init__(self, base_dir: str):
        self.output_dir = base_dir
        self.citations_validation_log = os.path.join(base_dir, "citations_validation.json")
        os.makedirs(self.output_dir, exist_ok=True)

    def safe_write(self, filename: str, content: str, mode: str = "a"):
        file_path = os.path.join(self.output_dir, filename)
        try:
            with open(file_path, mode, encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"文件操作失败 {file_path}: {e}")

    def save_json(self, filename: str, data: dict):
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def append_section(self, title: str, content: str):
        block = f"## {title}\n\n{content}\n\n"
        self.safe_write("report_streaming.md", block)

    def append_validation_logs(self, step_logs: list) -> None:
        """
        双轨日志：1. 写 JSON 供系统读取；2. 写 Markdown 供人类排查错误
        """
        if not step_logs:
            return

        try:
            import json
            import os
            from datetime import datetime
            
            # --- 1. 写入系统用的 JSON (保留原逻辑) ---
            existing_data = {}
            if os.path.exists(self.citations_validation_log):
                with open(self.citations_validation_log, "r", encoding="utf-8") as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        pass

            five_step_logs = existing_data.get("five_step_logs", [])
            if not isinstance(five_step_logs, list):
                five_step_logs = []

            five_step_logs.extend(step_logs)
            existing_data["five_step_logs"] = five_step_logs

            with open(self.citations_validation_log, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)

            # --- 2. 🔥 新增：写入给人类看的极致详细 Trace ---
            trace_msg = f"\n{'='*80}\n"
            trace_msg += f"🛡️ [CITATION VALIDATION START]\n"
            trace_msg += f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            for i, step in enumerate(step_logs, 1):
                trace_msg += f"\n{'-'*30} 步骤 {i}: {step.get('step', 'Unknown')} {'-'*30}\n"
                
                if 'input_text' in step:
                    trace_msg += f"📥 [输入文本]:\n{step['input_text']}\n\n"
                    
                if 'validation_result' in step:
                    trace_msg += f"🧠 [模型验证结果 (JSON)]:\n{json.dumps(step['validation_result'], ensure_ascii=False, indent=2)}\n\n"
                    
                if 'output_text' in step:
                    trace_msg += f"📤 [输出文本]:\n{step['output_text']}\n\n"
                    
                if 'error' in step and step['error']:
                    trace_msg += f"❌ [发生错误]: {step['error']}\n\n"
                    
            trace_msg += f"{'='*80}\n\n"
            
            # 统统追加到详细日志文件里
            self.safe_write("detailed_trace.log", trace_msg, mode="a")

        except Exception as e:
            error_msg = f"⚠️ [IO异常] 写入五步验证日志失败: {e}"
            print(error_msg)