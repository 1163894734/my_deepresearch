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


from smolagents import Tool
from smolagents.models import ChatMessage, MessageRole

class LongWriterTool(Tool):
    name = "long_writer"
    description = """
    A specialized tool for writing ultra-long reports (2,000 to 10,000+ words) based on instructions and research context.
    Use this tool whenever the user asks for a comprehensive report, deep research paper, or long-form article.
    You should provide the original instruction and all the gathered research context.
    """
    inputs = {
        "instruction": {
            "type": "string",
            "description": "The specific writing instruction (e.g., 'Write a 10,000-word history of the Roman Empire').",
        },
        "context": {
            "type": "string",
            "description": "The comprehensive research context or summaries gathered by the search agent.",
        }
    }
    output_type = "string"

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, instruction: str, context: str) -> str:
        # --- Step 1: Plan [cite: 103-105] ---
        # 1. 构造规划 Prompt
        plan_input = PLAN_PROMPT.format(user_instruction=instruction, context=context)
        
        # 2. 调用模型生成大纲
        # 注意：这里适配 LiteLLMModel 的调用方式
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": plan_input}])]
        plan_response = self.model(messages) 
        plan_text = plan_response.content
        
        # 3. 解析大纲步骤
        steps = [line.strip() for line in plan_text.strip().split('\n') if line.strip() and "Paragraph" in line]
        if not steps: 
            steps = [plan_text] # Fallback

        # --- Step 2: Write [cite: 134-136] ---
        full_content = ""
        
        # 串行生成每一段
        for step in steps:
            write_input = WRITE_PROMPT.format(
                user_instruction=instruction,
                plan=plan_text,
                written_text=full_content, # 关键：将已写内容传回 
                current_step=step
            )
            
            messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": write_input}])]
            
            # 论文建议 temperature=0.5，如果模型支持可以通过 kwargs 传入，默认亦可
            section_response = self.model(messages, temperature=0.5)
            
            # 拼接内容
            full_content += "\n\n" + section_response.content
            
        return full_content