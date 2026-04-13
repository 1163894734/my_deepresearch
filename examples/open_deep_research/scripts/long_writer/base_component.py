from __future__ import annotations
import json
from typing import Any, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from scripts.multi_agent.agent_context import PipelineContext

# 🔥 核心解耦 1：彻底删除对 long_writer_agent_v3 的导入，换成我们的新 Context


class JsonWorkflowComponent:
    """工作流组件基类：纯粹契约层。组件通过 Python dict 通信，JSON 序列化由本类静态方法统一处理。"""

    name: str = "base"
    description: str = ""
    input_format: Dict[str, Any] = {}
    output_format: Dict[str, Any] = {}

    @staticmethod
    def decode_input(message: str) -> Dict[str, Any]:
        """将输入解码为结构化字典"""
        raw = str(message or "").strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except Exception as e:
            raise ValueError(f"无效的组件输入 JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("组件输入必须是 JSON object。")
        return data

    @staticmethod
    def encode_output(data: Dict[str, Any]) -> str:
        """将结构化结果编码为 JSON 文本"""
        return json.dumps(data, ensure_ascii=False)

    # 🔥 核心解耦 2：签名从 agent 变为 context
    def run(self, context: "PipelineContext", payload: Dict[str, Any]) -> Dict[str, Any]:
        """执行组件的核心业务逻辑"""
        raise NotImplementedError

    def contract(self) -> Dict[str, Any]:
        """返回组件的输入输出契约定义"""
        return {
            "name": self.name,
            "description": self.description,
            "input_format": self.input_format,
            "output_format": self.output_format,
        }