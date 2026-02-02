import os
import re
import yaml
import time
from pathlib import Path
from smolagents import Tool, ChatMessage, MessageRole

class LongWriterTool(Tool):
    name = "long_writer"
    output_type = "string"
    
    # 定义输入参数
    inputs = {
        "instruction": {
            "type": "string", 
            "description": "写作指令，需包含字数、结构及参考文献数量要求。"
        },
        "context": {
            "type": "string", 
            "description": "搜索到的参考资料。"
        }
    }

    def __init__(self, model, skill_path_root=None, output_dir="outputs", **kwargs):
        super().__init__()
        self.model = model
        self.description = "智能长文写作工具，支持实时落盘保存及上下文备份。"
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 自动处理路径与资源加载
        if skill_path_root:
            self.skill_root = skill_path_root
            # 1. 动态读取描述文件
            desc_path = os.path.join(self.skill_root, "SKILL.md")
            if os.path.exists(desc_path):
                with open(desc_path, 'r', encoding='utf-8') as f:
                    self.description = f.read().strip()
            # 2. 动态读取 YAML 配置
            yaml_path = os.path.join(self.skill_root, "assets", "prompts.yaml")
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    self.prompts = yaml.safe_load(f)

    def forward(self, instruction: str, context: str) -> str:
        # ================= Step 0: 创建时间戳文件夹 =================
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 创建以时间戳命名的文件夹
        folder_name = f"report_{timestamp}"
        folder_path = os.path.join(self.output_dir, folder_name)
        
        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f">> [LongWriter] 创建输出文件夹: {folder_path}")
        except Exception as e:
            print(f">> [Warning] 文件夹创建失败: {e}")
            # 如果创建失败，使用输出目录作为备用
            folder_path = self.output_dir
        
        # ================= Step 1: 初始化输出文件 =================
        # 1. 定义报告文件 (Output)
        report_filename = f"report_{timestamp}.md"
        report_path = os.path.join(folder_path, report_filename)
        
        # 2. 定义上下文备份文件 (Input Context)
        context_filename = f"context_{timestamp}.md"
        context_path = os.path.join(folder_path, context_filename)

        print(f">> [LongWriter] 报告文件: {report_path}")
        print(f">> [LongWriter] 资料备份: {context_path}")

        # ================= Step 2: 备份上下文资料 =================
        try:
            with open(context_path, "w", encoding="utf-8") as f:
                f.write(f"# 写作任务资料备份\n\n")
                f.write(f"- 时间戳: {timestamp}\n")
                f.write(f"- 文件夹: {folder_name}\n")
                f.write(f"- 对应报告: [{report_filename}](./{report_filename})\n\n")
                f.write(f"## 1. 原始指令 (Instruction)\n\n{instruction}\n\n")
                f.write(f"## 2. 参考资料 (Context Summary)\n\n")
                f.write(context)
            print(f">> [LongWriter] ✅ 上下文资料已备份。")
        except Exception as e:
            print(f">> [Warning] 上下文备份失败: {e}")

        # ================= Step 3: 初始化报告文件 =================
        # 创建文件并写入标题/指令信息
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# 自动生成报告\n\n")
            f.write(f"> 生成时间：{timestamp}\n")
            f.write(f"> 原始指令：{instruction}\n")
            f.write(f"> 输出文件夹：{folder_name}\n")
            f.write(f"> 资料来源：[{context_filename}](./{context_filename})\n\n---\n\n")

        # ================= Step 4: Adaptive Plan (大纲) =================
        print(f">> [LongWriter] 正在规划大纲...")
        
        plan_input = self.prompts["plan_prompt"].format(
            instruction=instruction, 
            context_snippet=context[:5000]
        )
        
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": plan_input}])]
        plan_text = self.model(messages, temperature=0.5).content
        
        steps = self._parse_steps(plan_text)
        print(f">> [LongWriter] 大纲生成完成，共 {len(steps)} 步。")

        # 将大纲也写入报告文件备忘
        with open(report_path, "a", encoding="utf-8") as f:
            f.write("## 写作大纲\n")
            for s in steps:
                f.write(f"- {s}\n")
            f.write("\n---\n\n")

        # ================= Step 5: Global Bibliography (文献库) =================
        print(f">> [LongWriter] 正在构建全局文献库...")
        
        bib_input = self.prompts["bibliography_prompt"].format(
            instruction=instruction,
            context=context
        )
        
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": bib_input}])]
        global_bibliography_str = self.model(messages, temperature=0.2).content.strip()
        print(f">> [LongWriter] 文献库构建完成。")
        
        try:
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(f"\n## [Debug Info] 全局参考文献池\n> 以下是 AI 构建的可用参考文献列表，供写作时调用。\n\n")
                f.write(global_bibliography_str)
                f.write("\n\n---\n\n")
            print(f">> [LongWriter] ✅ 全局文献库已保存至文件。")
        except Exception as e:
            print(f">> [Warning] 文献库保存失败: {e}")

        # ================= Step 6: Rolling Write (流式写入) =================
        context_buffer = "" 
        
        for i, step in enumerate(steps):
            is_ref_step = ("参考文献" in step or "References" in step)
            print(f">> [Writing] ({i+1}/{len(steps)}) {step[:20]}... -> 写入硬盘")

            if is_ref_step:
                write_prompt = self.prompts["write_ref_prompt"].format(
                    global_bibliography_str=global_bibliography_str
                )
            else:
                write_prompt = self.prompts["write_section_prompt"].format(
                    instruction=instruction,
                    step=step,
                    global_bibliography_str=global_bibliography_str,
                    context=context,
                    previous_content=context_buffer[-2000:] if context_buffer else "（文章开头）"
                )

            messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": write_prompt}])]
            
            section_content = ""
            try:
                temp = 0.2 if is_ref_step else 0.4
                raw_content = self.model(messages, temperature=temp).content
                section_content = self._clean_output(raw_content, is_ref_step)
                
                with open(report_path, "a", encoding="utf-8") as f:
                    f.write(section_content + "\n\n")
                
                context_buffer += section_content + "\n\n"

            except Exception as e:
                error_msg = f"\n\n> [Error] 章节 '{step}' 写入失败: {e}\n\n"
                print(error_msg)
                with open(report_path, "a", encoding="utf-8") as f:
                    f.write(error_msg)

        # ================= Step 7: 返回结果 =================
        preview = context_buffer[:500].replace("\n", " ")
        return (
            f"✅ 长文写作已完成！\n"
            f"📁 输出文件夹：{folder_path}\n"
            f"📄 报告文件：{report_filename}\n"
            f"📚 资料备份：{context_filename}\n"
            f"📊 包含章节数：{len(steps)}\n"
            f"📝 内容预览：{preview}..."
        )

    def _parse_steps(self, plan_text):
        print(f"\n{'='*20} DEBUG: 模型原始大纲 {'='*20}")
        print(plan_text)
        print(f"{'='*50}\n")

        steps = []
        for line in plan_text.strip().split('\n'):
            line = line.strip()
            clean_content = line.replace("**", "").replace("__", "").strip()
            if (re.match(r'^(\d+(\.|、)|Step|STEP|第|Part|Chapter|\-|\*)', clean_content, re.IGNORECASE) 
                or "参考文献" in clean_content
                or "References" in clean_content):
                final_step = re.sub(r'^[\d\.\-*\s、]+', '', clean_content).strip()
                if len(final_step) > 2: 
                    steps.append(final_step)

        if len(steps) == 0:
            print(">> [Warning] 正则匹配失败，启用暴力提取模式...")
            steps = [line.strip() for line in plan_text.split('\n') if len(line.strip()) > 5]

        has_ref = any(("参考文献" in s or "References" in s) for s in steps)
        if steps and not has_ref:
            steps.append("参考文献 - 汇总列表")
            
        if not steps:
            steps = [
                "引言与背景",
                "核心概念分析",
                "技术细节深入",
                "应用与挑战",
                "结论",
                "参考文献 - 汇总列表"
            ]
            print(">> [Error] 模型未生成有效大纲，使用默认模板。")

        return steps

    def _clean_output(self, content, is_ref_step):
        content = re.sub(r"^(Sure|Here is|Okay|好的).*?\n", "", content, flags=re.IGNORECASE).strip()
        if not is_ref_step and "## 参考文献" in content:
            content = content.split("## 参考文献")[0].strip()
        return content