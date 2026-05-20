# utils/agent_helper.py
import os
import time
import logging

def setup_run_env(prefix="run", run_timestamp=None):
    """统一创建运行目录并配置日志"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_timestamp = run_timestamp or time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, "outputs", f"{prefix}_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    smol_logger = logging.getLogger("smolagents")
    smol_logger.setLevel(logging.INFO)
    smol_logger.handlers.clear() # 避免重复添加
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    smol_logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(os.path.join(run_dir, "agent_run.log"), encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    smol_logger.addHandler(file_handler)
    
    print(f"📁 专属运行目录已创建: {run_dir}")
    return run_dir, smol_logger

def export_agent_memory(agent, run_dir, filename="agent_thought_process.md"):
    """统一剥离出的 Agent 记忆输出逻辑"""
    memory_log_path = os.path.join(run_dir, filename)
    with open(memory_log_path, "w", encoding="utf-8") as f:
        f.write("# 🧠 Agent 思维链路深度解析\n\n")
        for i, step in enumerate(agent.memory.steps):
            f.write(f"## 🟢 [Step {i}]\n\n")
            if hasattr(step, 'duration') and step.duration:
                f.write(f"**⏱️ 本轮耗时:** `{step.duration:.2f} 秒`\n\n")
            if hasattr(step, 'model_output_message') and step.model_output_message:
                content = step.model_output_message.content
                if isinstance(content, list):
                    content = "\n".join([str(c.get('text', c)) for c in content if isinstance(c, dict)])
                f.write(f"### 💭 Agent 思考与动作\n{content}\n\n")
            if hasattr(step, 'observations') and step.observations:
                obs_text = str(step.observations)
                obs_text = obs_text[:2000] + "\n... [已截断]" if len(obs_text) > 2000 else obs_text
                f.write(f"### 👁️ 沙盒执行结果\n```text\n{obs_text}\n```\n\n")
            if hasattr(step, 'error') and step.error:
                f.write(f"### ❌ 报错信息\n```python\n{step.error}\n```\n\n")
            f.write("---\n\n")