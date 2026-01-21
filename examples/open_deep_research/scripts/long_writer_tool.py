# # 论文 Page 3-4 [cite: 126-132]
# PLAN_PROMPT = """
# I need you to help me break down the following long-form writing instruction into multiple subtasks.
# Each subtask will guide the writing of one paragraph in the essay, and should include the main points and word count requirements for that paragraph.

# The writing instruction is as follows:
# {user_instruction}

# Context/Research Material:
# {context}

# Please break it down in the following format, with each subtask taking up one line:
# Paragraph 1 - Main Point: [Describe the main point of the paragraph, in detail] - Word Count: [Word count requirement, e.g., 400 words]
# Paragraph 2 - Main Point: [Describe the main point of the paragraph, in detail] - Word Count: [word count requirement, e.g. 1000 words].

# Make sure that each subtask is clear and specific, and that all subtasks cover the entire content of the writing instruction.
# Do not split the subtasks too finely; each subtask's paragraph should be no less than 200 words and no more than 1000 words.
# Do not output any other content.
# """

# # 论文 Page 4 [cite: 138-147]
# WRITE_PROMPT = """
# You are an excellent writing assistant. I will give you an original writing instruction and my planned writing steps.
# I will also provide you with the text I have already written.
# Please help me continue writing the next paragraph based on the writing instruction, writing steps, and the already written text.

# Writing instruction:
# {user_instruction}

# Writing steps:
# {plan}

# Already written text:
# {written_text}

# Please integrate the original writing instruction, writing steps, and the already written text, and now continue writing {current_step} for me.
# If needed, you can add a small subtitle at the beginning.
# Remember to only output the paragraph you write, without repeating the already written text.
# """

import re
from smolagents import Tool
from smolagents.models import ChatMessage, MessageRole

class LongWriterTool(Tool):
    name = "long_writer"
    description = "智能长文写作工具。具备全局文献管理能力，防止引用重复或遗漏。"
    inputs = {
        "instruction": {"type": "string", "description": "写作指令，需包含字数、结构及参考文献数量要求。"},
        "context": {"type": "string", "description": "搜索到的参考资料。"}
    }
    output_type = "string"

    def __init__(self, model, max_context_tokens=100000):
        super().__init__()
        self.model = model
        self.max_context_tokens = max_context_tokens

    def forward(self, instruction: str, context: str) -> str:
        # ==============================================================================
        # Step 1: Adaptive Plan (大纲生成)
        # ==============================================================================
        print(f">> [LongWriter] 正在规划大纲...")
        
        plan_input = f"""
        【角色】专业学术写作顾问
        【任务】根据用户指令制定写作大纲。
        【指令】"{instruction}"
        【资料片段】{context[:5000]}...

        【要求】
        1. 根据指令要求的篇幅规划步骤（短文3-5步，长文8-15步）。
        2. 格式：每一行一个步骤，包含[核心论点]和[预估字数]。
        3. **最后一步必须是**：STEP: 参考文献 - 汇总列表
        
        【输出示例】
        STEP: 第1部分 - [引言] - [500字]
        ...
        STEP: 参考文献 - 汇总列表
        """
        
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": plan_input}])]
        plan_text = self.model(messages, temperature=0.5).content
        
        steps = []
        for line in plan_text.strip().split('\n'):
            line = line.strip()
            # 兼容多种格式的步骤解析
            if line and (line.startswith("STEP:") or "第" in line or re.match(r"^\d+\.", line) or "参考文献" in line):
                clean_line = line.replace("*", "").replace("#", "").strip()
                steps.append(clean_line)
        
        # 保底逻辑：如果模型没生成参考文献步骤，手动强制追加
        if steps and not any("参考文献" in s or "References" in s for s in steps[-1:]):
            steps.append("STEP: 参考文献 - 汇总列表")

        print(f">> [LongWriter] 大纲生成完成，共 {len(steps)} 步。")

        # ==============================================================================
        # Step 1.5: Global Bibliography Generation (核心修复：自适应数量 & 全局去重)
        # ==============================================================================
        print(f">> [LongWriter] 正在构建全局文献库 (提取所有可用来源)...")
        
        # 策略：告诉模型用户的具体要求，强制它“榨干”Context
        bib_prompt = f"""
        【任务目标】
        基于参考资料，为用户的写作任务构建一个**完整、详尽**的参考文献库。
        
        【用户原始指令】
        "{instruction}"
        (注意：如果用户要求了参考文献的数量，或者要求“全面综述”，请务必尽可能多地提取资料中的文献。)

        【参考资料】
        {context} 
        
        【提取要求】
        1. **穷尽原则**：请仔细扫描资料，提取所有提到的论文、书籍、报告或网页。不要遗漏。
        2. **去重处理**：如果资料中同一篇论文出现了多次，只保留一个。
        3. **编号格式**：分配唯一 ID，格式为 [1], [2], [3]...
        4. **内容格式**：
           [ID] 作者/机构. 标题. (年份/来源)
        5. **数量控制**：
           - 如果用户未指定数量，至少提取 10-20 条（如果资料足够）。
           - 如果用户要求了数量（如“至少30条”），请全力满足。
        
        【输出】仅输出文献列表，不要包含任何对话开头。
        """
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": bib_prompt}])]
        # 温度调低，保证提取的准确性
        global_bibliography_str = self.model(messages, temperature=0.2).content.strip()
        
        # 简单统计一下提取了多少，打印日志方便调试
        count = global_bibliography_str.count("[")
        print(f">> [LongWriter] 已构建文献库，包含约 {count} 条文献。")

        # ==============================================================================
        # Step 2: Rolling Write (分段写作 & 防重复)
        # ==============================================================================
        full_content = ""
        outline_reference = "\n".join(steps) 

        for i, step in enumerate(steps):
            is_ref_step = ("参考文献" in step or "References" in step)
            
            print(f">> [Writing] ({i+1}/{len(steps)}) 正在处理: {step[:20]}...")

            if is_ref_step:
                # --- 分支 A：参考文献段落 ---
                # 直接输出 Step 1.5 生成的列表，绝不重新生成，防止幻觉和不一致
                write_prompt = f"""
                任务：输出文章最后的参考文献列表。
                
                【指令】：
                直接输出以下我们整理好的全局文献库，不要做任何删减或修改。
                
                全局文献库：
                {global_bibliography_str}
                
                【格式要求】：
                1. 使用一级或二级标题： "## 参考文献"
                2. 直接列出内容。
                """
            else:
                # --- 分支 B：正文段落 ---
                # 核心修复：加入【负面约束】，防止模型在最后一段正文里自己把参考文献写了
                write_prompt = f"""
                你正在撰写一篇长文。
                
                【用户总指令】：{instruction}
                【当前写作任务】：撰写章节——{step}
                
                【全局文献库（供引用标注使用）】：
                {global_bibliography_str}
                
                【参考资料原文】：
                {context}
                
                【前文脉络】：
                {full_content[-2000:] if full_content else "（文章开头）"}
                
                【严格写作要求】：
                1. **标题格式**：
                - 必须使用 Markdown 二级标题（## ）。
                - 标题内容仅保留核心主题词，**去掉**“第X部分”、“STEP”、“字数”以及方括号等符号。
                - 例如：任务为“STEP: 第1部分 - [引言] - [500字]”，标题应写为“## 引言”。
                2. **引用标注**：文中引用观点时，**必须**使用文献库中对应的编号（如 [1], [5]）。禁止编造不在库里的编号。
                3. **内容深度**：内容要充实，逻辑连贯。
                4. **禁止重复输出**：
                   - **绝对禁止**在段落末尾输出“参考文献列表”！
                   - 参考文献将由专门的后续步骤生成，这里只写正文！
                   - 不要写“综上所述”、“我们将在下一章”等废话。
                """

            messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": write_prompt}])]
            try:
                # 正文写作温度稍高，保证流畅度
                temp = 0.2 if is_ref_step else 0.4
                section_content = self.model(messages, temperature=temp).content
                
                # 清洗模型可能输出的废话
                section_content = re.sub(r"^(Sure|Here is|Okay|好的).*?\n", "", section_content, flags=re.IGNORECASE).strip()
                
                # 二次检查：如果正文段落（非Ref步骤）里包含了大量的 "[1] 作者..." 列表，说明模型没听话，强制剪裁
                if not is_ref_step and "## 参考文献" in section_content:
                     section_content = section_content.split("## 参考文献")[0].strip()
                
                full_content += "\n\n" + section_content
            except Exception as e:
                print(f">> [Error] 写作失败: {e}")
                full_content += f"\n\n[Section Error]"

        return full_content