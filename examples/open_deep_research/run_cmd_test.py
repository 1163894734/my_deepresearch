import os
import json
import logging
from utils import common_utils
from smolagents import CodeAgent, CustomAgent
# 引入所有所需工具
from scripts.custom_tools import PDFVisionParserTool

# =========================
# 1. 基础配置与日志
# =========================
base_dir = os.path.dirname(os.path.abspath(__file__))
# ⚠️ 注意替换目录
run_dir = "/Users/wangchao/project/my_deepresearch/examples/outputs/deep_research_20260423_101758"

def main():
    model = common_utils.ModelProvider.get_model()

    # =========================
    # 3. 实例化最高总管
    # =========================
    writer_director_agent = CodeAgent(
        name="writer_director_agent",
        description="最高总管。负责解析大纲并调用底层的自动组装引擎。",
        tools=[PDFVisionParserTool()],
        model=model,
        additional_authorized_imports=["json"],
        instructions="""
        调用工具tool_vision_parse(rundir+"/pdfs/paper_1ff46668.pdf",rundir)
        """
    )

    writer_director_agent.state["run_dir"] = run_dir
    writer_director_agent.run("运行")


if __name__ == "__main__":
    main()