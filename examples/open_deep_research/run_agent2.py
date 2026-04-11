import os
import json
import time
import logging
from smolagents import CodeAgent
from scripts.skill_loader import load_skills_from_directory
from smolagents.models import OpenAIModel
import utils.common_utils as common_utils

# =========================
# 1. 初始化运行目录与全量日志
# =========================
base_dir = os.path.dirname(os.path.abspath(__file__))
run_timestamp = time.strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(base_dir, "outputs", f"identification_{run_timestamp}")
os.makedirs(run_dir, exist_ok=True)

# 捕获 smolagents 的原生日志
smol_logger = logging.getLogger("smolagents")
smol_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join(run_dir, "agent_run.log"), encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
smol_logger.addHandler(file_handler)

def main():
    print(f"📁 本次任务专属目录已创建: {run_dir}")
    smol_logger.info("🚀 启动前沿技术识别 Agentic Workflow...")

    # =========================
    # 2. 加载技能与模型
    # =========================
    model = common_utils.ModelProvider.get_model()
    skills = load_skills_from_directory("skills", model=model)

    GENERIC_INSTRUCTIONS = """
    你是一个高度自主的 AI 助手。你可以编写并在沙盒中运行 Python 代码来处理数据。
    当你遇到复杂的垂直业务请求时，请先检查工具列表中是否有对应的 SOP 或指南工具，并优先调用它们来了解你应该怎么做。
    """

    agent = CodeAgent(
        tools=skills,
        model=model,
        instructions=GENERIC_INSTRUCTIONS,
        additional_authorized_imports=["json", "time"]
    )

    # =========================
    # 3. 注入数据与环境变量
    # =========================
    test_file = "scripts/test_payload.json"
    if os.path.exists(test_file):
        with open(test_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {"documents": [], "domain": "AI Agent"}
        smol_logger.warning("未找到测试数据，使用空数据流。")

    agent.state["raw_documents"] = payload.get("documents", [])
    agent.state["target_domain"] = payload.get("domain", "AI Agent")
    # 【关键】把路径注入给 Agent，让它传给 Save Tool
    agent.state["run_dir"] = run_dir 

    # =========================
    # 4. 执行任务
    # =========================
    try:
        result = agent.run("请帮我识别并分析一下这批数据中关于 AI Agent 的前沿技术。")
        # ========== ✨ 新增：转存 Agent 的完整思维链路 (Memory) [排版优化版] ==========
        memory_log_path = os.path.join(run_dir, "agent_thought_process.md")
        with open(memory_log_path, "w", encoding="utf-8") as f:
            f.write("# 🧠 Agent 思维链路深度解析\n\n")
            
            for i, step in enumerate(agent.memory.steps):
                f.write(f"## 🟢 [Step {i}]\n\n")
                
                # 1. 打印耗时 (如果有)
                if hasattr(step, 'duration') and step.duration:
                    f.write(f"**⏱️ 本轮耗时:** `{step.duration:.2f} 秒`\n\n")
                
                # 2. 提取大模型的真实输出（包含它的“内心独白”和它写的 Python 代码）
                if hasattr(step, 'model_output_message') and step.model_output_message:
                    content = step.model_output_message.content
                    # smolagents 的 content 可能是列表，需要拼接
                    if isinstance(content, list):
                        content = "\n".join([str(c.get('text', c)) for c in content if isinstance(c, dict)])
                    
                    f.write("### 💭 Agent 思考与动作 (Thought & Action)\n")
                    f.write(f"{content}\n\n")
                
                # 3. 提取沙盒执行的返回结果
                if hasattr(step, 'observations') and step.observations:
                    f.write("### 👁️ 沙盒执行结果 (Observation)\n")
                    # 限制过长的控制台输出，防止刷屏 (取前2000个字符)
                    obs_text = str(step.observations)
                    if len(obs_text) > 2000:
                        obs_text = obs_text[:2000] + "\n... [输出过长，已截断]"
                    f.write(f"```text\n{obs_text}\n```\n\n")
                
                # 4. 提取代码报错信息 (如果有)
                if hasattr(step, 'error') and step.error:
                    f.write("### ❌ 报错信息 (Error)\n")
                    f.write(f"```python\n{step.error}\n```\n\n")
                
                f.write("---\n\n")
        # =================================================================
        smol_logger.info("✅ 任务圆满完成。")
        print("\n================ 最终研报 ================\n")
        print(result)
    except Exception as e:
        smol_logger.error(f"❌ 执行过程中发生致命错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()