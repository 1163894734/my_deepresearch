# 论文 Page 3-4 [cite: 126-132]
PLAN_PROMPT = """
I need you to help me break down the following long-form writing instruction into multiple subtasks.
Each subtask will guide the writing of one paragraph in the essay, and should include the main points and word count requirements for that paragraph.

The writing instruction is as follows:
{user_instruction}

Context/Research Material:
{context}

Please break it down in the following format, with each subtask taking up one line:
Paragraph 1 - Main Point: [Describe the main point of the paragraph, in detail] - Word Count: [Word count requirement, e.g., 400 words]
Paragraph 2 - Main Point: [Describe the main point of the paragraph, in detail] - Word Count: [word count requirement, e.g. 1000 words].

Make sure that each subtask is clear and specific, and that all subtasks cover the entire content of the writing instruction.
Do not split the subtasks too finely; each subtask's paragraph should be no less than 200 words and no more than 1000 words.
Do not output any other content.
"""

# 论文 Page 4 [cite: 138-147]
WRITE_PROMPT = """
You are an excellent writing assistant. I will give you an original writing instruction and my planned writing steps.
I will also provide you with the text I have already written.
Please help me continue writing the next paragraph based on the writing instruction, writing steps, and the already written text.

Writing instruction:
{user_instruction}

Writing steps:
{plan}

Already written text:
{written_text}

Please integrate the original writing instruction, writing steps, and the already written text, and now continue writing {current_step} for me.
If needed, you can add a small subtitle at the beginning.
Remember to only output the paragraph you write, without repeating the already written text.
"""

import re
from smolagents import Tool
from smolagents.models import ChatMessage, MessageRole

class LongWriterTool(Tool):
    name = "long_writer"
    description = "智能长文写作工具。根据指令复杂度自动规划大纲（从短文到长篇论文），并分段生成内容。"
    inputs = {
        "instruction": {"type": "string", "description": "写作指令，需包含字数和结构要求。"},
        "context": {"type": "string", "description": "搜索到的参考资料。"}
    }
    output_type = "string"

    def __init__(self, model, max_context_tokens=100000):
        super().__init__()
        self.model = model
        self.max_context_tokens = max_context_tokens

    def _estimate_tokens(self, text):
        return len(text) * 1.5 

    def forward(self, instruction: str, context: str) -> str:
        # --- Step 1: Adaptive Plan (自适应大纲生成) ---
        print(f">> [LongWriter] 正在分析任务并规划大纲 (Context长度: {len(context)})...")
        
        # 【修改点 1】: 根据 context 长度和 instruction 动态调整 Prompt 语气
        # 移除硬编码的 "10-15步"，改为由模型根据指令判断
        plan_input = f"""
        【角色设定】
        你是一位专业的学术写作顾问。请根据用户指令和参考资料，制定一份合理的**写作大纲**。

        【用户指令 (User Instruction)】
        "{instruction}"

        【参考资料摘要 (Context)】
        {context[:5000]}... (略)

        【大纲规划要求 - 请严格遵守】
        1. **结构自适应**：
           - 如果用户要求“短文”、“几段话”或字数较少（<2000字），请只规划 **3-5 个** 核心步骤（如：引言、核心段落1、核心段落2、结论）。
           - 如果用户要求“万字长文”、“调研报告”、“详细报告”或“书籍”，请规划 **8-15 个** 详细章节。
        2. **逻辑连贯**：步骤之间要有逻辑递进关系。
        3. **基于事实**：不要规划参考资料中完全不存在的内容章节。
        4. **格式规范**：每一行输出一个步骤，包含 [核心论点] 和 [预估字数]，如果用户有字数要求，按照用户的字数要求规划每段的 [预估字数]。

        【输出格式示例】
        STEP: 第1部分 - [引言：背景介绍] - [300字]
        STEP: 第2部分 - [核心观点A] - [500字]
        ...
        STEP: 参考文献 - 整理引用列表
        """
        
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": plan_input}])]
        plan_text = self.model(messages,temperature=0.5).content
        
        # --- Step 1.5: Parsing (保持原有的鲁棒解析，但增加清洗) ---
        steps = []
        for line in plan_text.strip().split('\n'):
            line = line.strip()
            # 过滤掉空的或无关的行，只保留像步骤的行
            if line and (line.startswith("STEP:") or "第" in line or re.match(r"^\d+\.", line)):
                clean_line = line.replace("*", "").replace("#", "").strip()
                steps.append(clean_line)
        
        # 如果模型最后忘了加参考文献，手动补上
        if steps and not any("参考文献" in s or "References" in s for s in steps[-1:]):
            steps.append("STEP: 参考文献 - 整理引用列表")

        print(f">> [LongWriter] 生成了 {len(steps)} 个写作步骤:")
        for idx, s in enumerate(steps): 
            print(f"   {idx+1}. {s}")

        # --- Step 2: Rolling Write (分段写作 - 保持原有逻辑) ---
        full_content = ""
        outline_reference = "\n".join(steps) 

        for i, step in enumerate(steps):
            # 判断是否是参考文献步骤
            is_ref_step = ("参考文献" in step or "References" in step)
            
            print(f">> [Writing] ({i+1}/{len(steps)}) {step[:30]}...")

            if is_ref_step:
                # 【修改点】：参考文献 Prompt 增加标题和编号要求
                write_prompt = f"""
                任务：根据以下资料整理本文引用的参考文献列表。
                资料：{context[-20000:]} 
                
                【格式严重要求】：
                1. **必须**以 "## 参考文献" 作为标题开头。
                2. **必须**使用数字编号格式，例如：
                   [1] Vaswani, A., et al. (2017). Attention Is All You Need.
                   [2] Devlin, J., et al. (2019). BERT...
                3. 仅列出文中实际可能用到的文献，不要凭空捏造。
                4. 直接输出内容，不要包含“好的”、“清单如下”等废话。
                5. 参考文献的所有编号必须是连续的整数，从1开始。任何情况下都不能出现编号空缺，例如，2后面必须是3，不能直接跳到4或任何其他数字。
                """
            else:
                # 正文写作 Prompt (保持原有逻辑，确保不越界)
                write_prompt = f"""
                总任务：{instruction}
                完整大纲：\n{outline_reference}
                
                当前步骤：【{step}】
                
                参考资料：{context[:30000]}
                前文脉络：{full_content[-2000:] if full_content else "（文章开头）"}
                
                写作要求：
                1. 严格按照“当前步骤”的主题写作。
                2. 使用 Markdown 格式（使用 ## 或 ### 标题）。
                3. 如果引用了观点，可以尝试加入标注（如 [1]），但不需要强制对应具体的参考文献列表（因为列表是最后生成的），保持逻辑通顺即可。
                4. 直接输出内容。
                5. 请勿输出前文已经输出过的段落或内容。
                6. 标题严格安装“当前步骤”中描述的进行书写，但不需要在标题中体现字数要求。
                """

            # 执行生成
            messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": write_prompt}])]
            try:
                section_content = self.model(messages,temperature=0.5).content
                # 简单清洗
                section_content = re.sub(r"^(Sure|Here is|Okay).*?\n", "", section_content, flags=re.IGNORECASE).strip()
                full_content += "\n\n" + section_content
            except Exception as e:
                print(f">> [Error] 写作失败: {e}")
                full_content += f"\n\n[Section Error]"

        return full_content