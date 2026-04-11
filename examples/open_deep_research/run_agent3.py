import argparse
import os
import json
import time
import logging
from dotenv import load_dotenv

# 引入 smolagents 核心组件
from smolagents.default_tools import DuckDuckGoSearchTool
from utils import common_utils
from smolagents import CodeAgent
from smolagents.models import OpenAIModel

# 引入通用的技能加载器
from scripts.skill_loader import load_skills_from_directory

load_dotenv(override=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question", type=str, nargs="?", default="具身智能", help="需要解读的技术名词"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    target_term = args.question

    # =========================
    # 1. 初始化运行目录与全量日志
    # =========================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, "outputs", f"interpretation_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 捕获 smolagents 的原生日志并写入文件
    smol_logger = logging.getLogger("smolagents")
    smol_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(run_dir, "agent_run.log"), encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    smol_logger.addHandler(file_handler)

    print(f"📁 本次任务专属目录已创建: {run_dir}")
    smol_logger.info(f"🚀 启动技术名词解读 Agentic Workflow... 目标名词: {target_term}")

    # =========================
    # 2. 加载技能与模型
    # =========================
    model = common_utils.ModelProvider.get_model()
    
    # 动态加载所有的 skills，包括工具和 SOP
    # （注：你需要将以前的学术检索、Web检索等封装成独立的 Tool 放入 skills 目录下）
    skills = load_skills_from_directory("skills", model=model)

    GENERIC_INSTRUCTIONS = """
    你是一个高度自主的 AI 助手。你可以编写并在沙盒中运行 Python 代码来处理数据和调用各类工具。
    当你遇到复杂的垂直业务请求（如“技术名词解读”）时，请先检查工具列表中是否有对应的 SOP 或指南工具，并优先调用它们来了解你应该按什么标准流程执行。
    """
    web_search_tool = DuckDuckGoSearchTool()
    web_search_tool.name = "web_search"
    # 实例化基于代码的 Agent
    agent = CodeAgent(
        tools=skills+[web_search_tool],
        model=model,
        instructions=GENERIC_INSTRUCTIONS,
        additional_authorized_imports=["json", "time", "os", "re", "concurrent.futures"]
    )

    # =========================
    # 3. 注入数据与环境变量
    # =========================
    # 将目标名词和目录注入到 state 中，方便 Agent 在编写 Python 代码时直接读取
    agent.state["target_term"] = target_term
    agent.state["run_dir"] = run_dir 

    # =========================
    # 4. 执行任务
    # =========================
    try:
        # 向 Agent 下发指令，Agent 会自主寻找 SOP 并执行
        prompt = f"请帮我执行【技术名词解读】任务。需要解读的目标概念是：{target_term}。请查阅相关SOP（如果存在），综合调用学术检索与网络搜索工具，最终生成百科版、专报版、科普版三版报告并保存到我指定的 run_dir 中。"
        
        result = agent.run(prompt)
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
        print("\n================ 最终解读报告 ================\n")
        print(result)
        
    except Exception as e:
        smol_logger.error(f"❌ 执行过程中发生致命错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()