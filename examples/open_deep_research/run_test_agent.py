import os
import json
import time
import logging
from smolagents import CodeAgent
from scripts.skill_loader import load_skills_from_directory
from smolagents.models import OpenAIModel
import utils.common_utils as common_utils
from scripts.custom_tools import MdToWordTool, SaveFileTool, GetVariableTool, SetVariableTool, LoadFileTool, GenerateAbstractTool, GenerateBibliographyTool, GenerateConclusionTool, ParseJsonTool, FlattenOutlineTool

run_dir = "/Users/wangchao/project/my_deepresearch/examples/outputs/deep_research_20260511_205655"

def main():
    
    model = common_utils.ModelProvider.get_model()
    agent = CodeAgent(
        tools=[MdToWordTool(),SaveFileTool(), GetVariableTool(), SetVariableTool(), LoadFileTool(), GenerateAbstractTool(), GenerateBibliographyTool(), GenerateConclusionTool(), ParseJsonTool(), FlattenOutlineTool()],
        model=model,
        instructions="使用提供的工具将run_dir下的final_academic_report_final.md文件转换为word文档，run_dir变量已注入环境，直接使用即可。",
        additional_authorized_imports=["json", "time"]
    )

    
    agent.state["run_dir"] = run_dir
    agent.run("干活")
if __name__ == "__main__":
    main()