from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from ..long_writer_agent_v3 import LongWriterAgent
    except Exception:
        from long_writer_agent_v3 import LongWriterAgent

try:
    from .base_component import JsonWorkflowComponent
    from .cli_debugger import run_component_cli
    from .section_writing_service import SectionWritingService
except ImportError:
    from base_component import JsonWorkflowComponent
    from cli_debugger import run_component_cli
    from section_writing_service import SectionWritingService


class AbstractWritingComponent(JsonWorkflowComponent):
    """摘要撰写组件。

    输入 JSON 示例:
    {
        "section": {"title": "摘要", "goal": "凝练核心发现", "word_count_target": 300},
        "conclusion_text": "（结论内容）",
        "full_text": "（全文内容）",
        "task": "（用户任务描述）"
    }

    输出 JSON 示例:
    {
        "content": "摘要正文..."
    }
    """

    name = "abstract_writing"
    description = "摘要撰写组件：输入摘要章节配置与结论文本，输出摘要终稿。"
    input_format = {
        "section": "Dict",
        "conclusion_text": "str",
        "full_text": "str",
        "task": "str",
    }
    output_format = {
        "content": "str",
    }

    def run(self, agent: "LongWriterAgent", payload: dict) -> dict:
        """
        主要作用：执行当前组件的主流程，处理输入并返回结构化输出。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - payload (dict): 组件输入字典，通常包含任务、章节信息、检索材料、引用元数据或其他工作流中间结果。

        返回值：
        - dict：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 读取方法所需输入并做必要预处理。
        - 执行该方法对应的核心业务逻辑。
        - 返回结果或通过副作用更新状态、日志和文件。
        """

        final = SectionWritingService.write_abstract_content_with_json(agent, payload)
        return {"content": str(final)}


def main() -> int:
    """
    主要作用：执行 main 相关逻辑。

    输入参数：
    - 无：该方法不接收显式业务参数。

    返回值：
    - int：返回状态码、计数值或其他数值结果。

    实现逻辑：
    - 读取方法所需输入并做必要预处理。
    - 执行该方法对应的核心业务逻辑。
    - 返回结果或通过副作用更新状态、日志和文件。
    """

    if len(sys.argv) <= 1:
        workspace_root = Path(__file__).resolve().parents[4]
        long_text_path = workspace_root / "long_text.md"
        full_text = long_text_path.read_text(encoding="utf-8") if long_text_path.exists() else ""
        payload_json_text = json.dumps(
            {
                "input": {
                    "section": {"title": "摘要", "goal": "凝练核心发现", "word_count_target": 300},
                    "conclusion_text": "（结论内容）",
                    "full_text": full_text,
                    "task": "撰写一篇关于大模型领域主要文献与近期进展的完整中文调研报告。",
                }
            },
            ensure_ascii=False,
        )
        return run_component_cli(
            AbstractWritingComponent(),
            argv=["--input-json", payload_json_text],
        )
    return run_component_cli(AbstractWritingComponent())


if __name__ == "__main__":
    raise SystemExit(main())
