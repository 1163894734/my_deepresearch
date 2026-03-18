from __future__ import annotations

import importlib
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    from ..long_writer_agent_v3 import LongWriterAgent


class JsonWorkflowComponent:
    """工作流组件基类：组件通过 Python dict 通信，JSON 序列化由本类静态方法统一处理。"""

    name: str = "base"
    description: str = ""
    input_format: Dict[str, Any] = {}
    output_format: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # JSON 编解码工具（集中在此，主流程/agent 无需关心序列化）              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def decode_input(message: str) -> Dict[str, Any]:
        """
        主要作用：把 CLI 或协议输入解码为结构化字典。

        输入参数：
        - message (str): 待解析的输入消息文本，通常来自组件 CLI、模型输出或结构化协议消息。

        返回值：
        - Dict[str, Any]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 扫描输入中的关键模式、字段或噪声片段。
        - 完成清洗、规范化、回填或结构化整理。
        - 返回更稳定、可复用的中间结果。
        """
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
        """
        主要作用：把结构化结果编码为 JSON 文本。

        输入参数：
        - data (Dict[str, Any]): 结构化数据字典，表示中间状态、解析结果、配置或组件返回值。

        返回值：
        - str：返回处理后的文本、提示词、章节内容或格式化字符串。

        实现逻辑：
        - 读取方法所需输入并做必要预处理。
        - 执行该方法对应的核心业务逻辑。
        - 返回结果或通过副作用更新状态、日志和文件。
        """
        return json.dumps(data, ensure_ascii=False)

    def run(self, agent: "LongWriterAgent", payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        主要作用：执行当前组件的主流程，处理输入并返回结构化输出。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。
        - agent ("LongWriterAgent"): 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
        - payload (Dict[str, Any]): 组件输入字典，通常包含任务、章节信息、检索材料、引用元数据或其他工作流中间结果。

        返回值：
        - Dict[str, Any]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 读取方法所需输入并做必要预处理。
        - 执行该方法对应的核心业务逻辑。
        - 返回结果或通过副作用更新状态、日志和文件。
        """
        raise NotImplementedError

    def contract(self) -> Dict[str, Any]:
        """
        主要作用：返回组件的输入输出契约定义。

        输入参数：
        - self: 当前对象实例，用于访问成员配置、运行状态、缓存和协作服务。

        返回值：
        - Dict[str, Any]：返回结构化字典结果，便于后续工作流阶段继续消费。

        实现逻辑：
        - 读取方法所需输入并做必要预处理。
        - 执行该方法对应的核心业务逻辑。
        - 返回结果或通过副作用更新状态、日志和文件。
        """

        return {
            "name": self.name,
            "description": self.description,
            "input_format": self.input_format,
            "output_format": self.output_format,
        }


def _import_long_writer_agent():
    """
    主要作用：动态导入 LongWriterAgent，避免静态依赖导致循环引用。

    输入参数：
    - 无：该方法不接收显式业务参数。

    返回值：
    - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

    实现逻辑：
    - 整理当前阶段的核心信息。
    - 按照既定格式写入日志、文件或状态对象。
    - 保证运行过程可回溯、可排障、可复现。
    """
    try:
        from ..long_writer_agent_v3 import LongWriterAgent

        return LongWriterAgent
    except Exception:
        scripts_dir = Path(__file__).resolve().parent.parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from long_writer_agent_v3 import LongWriterAgent

        return LongWriterAgent


def _resolve_factory(factory_path: str) -> Callable[..., Any]:
    """
    主要作用：根据导入路径字符串解析工厂函数。

    输入参数：
    - factory_path (str): 工厂函数的导入路径字符串。

    返回值：
    - Callable[..., Any]：返回该方法的主要输出结果。

    实现逻辑：
    - 读取待校验输入并应用当前业务约束。
    - 执行验证、判定、拒绝或映射逻辑。
    - 返回验证结果，或同步更新状态与日志。
    """
    if ":" not in factory_path:
        raise ValueError("`agent_config.factory` 必须是 'package.module:callable' 格式")

    module_name, callable_name = factory_path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, callable_name, None)
    if factory is None or not callable(factory):
        raise ValueError(f"无法解析可调用工厂: {factory_path}")
    return factory


def _normalize_agent_output_paths(agent: Any, init_cwd: Path, target_cwd: Path) -> None:
    """
    主要作用：修正组件 CLI 场景下代理内部的输出路径。

    输入参数：
    - agent (Any): 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
    - init_cwd (Path): 构建代理前的当前工作目录。
    - target_cwd (Path): 组件 CLI 希望切换到的工作目录。

    返回值：
    - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

    实现逻辑：
    - 扫描输入中的关键模式、字段或噪声片段。
    - 完成清洗、规范化、回填或结构化整理。
    - 返回更稳定、可复用的中间结果。
    """
    path_attrs = [
        "_output_dir",
        "_output_file",
        "_log_file",
        "_reference_trace_file",
        "_citations_validation_log",
        "_tagged_search_log_file",
    ]

    for attr in path_attrs:
        raw = getattr(agent, attr, None)
        if not raw:
            continue
        p = Path(str(raw))
        abs_path = p if p.is_absolute() else (init_cwd / p)
        try:
            rel_path = abs_path.relative_to(target_cwd)
            setattr(agent, attr, str(rel_path))
        except Exception:
            setattr(agent, attr, os.path.relpath(str(abs_path), start=str(target_cwd)))

    output_dir = getattr(agent, "_output_dir", None)
    if output_dir:
        Path(str(output_dir)).mkdir(parents=True, exist_ok=True)


def build_agent_for_component_cli(agent_config: Optional[Dict[str, Any]] = None):
    """
    主要作用：为组件 CLI 调试构建可运行代理。

    输入参数：
    - agent_config (Optional[Dict[str, Any]]): 该参数用于承载 `agent_config` 相关的业务上下文或控制信息。

    返回值：
    - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

    实现逻辑：
    - 读取当前上下文中的关键字段。
    - 按既定模板和业务规则拼装输入结构。
    - 返回下游阶段可直接消费的提示词、映射或载荷。
    """
    config = dict(agent_config or {})

    factory_path = str(config.get("factory", "")).strip()
    factory_kwargs = config.get("factory_kwargs", {})
    if factory_path:
        if not isinstance(factory_kwargs, dict):
            raise ValueError("`agent_config.factory_kwargs` 必须是 JSON object")
        agent = _resolve_factory(factory_path)(**factory_kwargs)
    else:
        runtime_model_id = str(config.get("model_id", "o1")).strip() or "o1"
        previous_cwd = Path.cwd()
        init_cwd = previous_cwd
        try:
            open_deep_research_dir = Path(__file__).resolve().parent.parent.parent
            init_cwd = open_deep_research_dir
            os.chdir(str(open_deep_research_dir))
            try:
                run_module = importlib.import_module("examples.open_deep_research.run")
            except Exception:
                if str(open_deep_research_dir) not in sys.path:
                    sys.path.insert(0, str(open_deep_research_dir))
                run_module = importlib.import_module("run")

            create_agent = getattr(run_module, "create_agent", None)
            if create_agent is None or not callable(create_agent):
                raise RuntimeError("run.py 中未找到 create_agent(model_id=...) 工厂")

            manager_agent = create_agent(model_id=runtime_model_id)
            if getattr(manager_agent, "name", "") == "long_writer_agent":
                agent = manager_agent
            else:
                managed = getattr(manager_agent, "managed_agents", None)
                agent = None
                if isinstance(managed, dict):
                    agent = managed.get("long_writer_agent")
                elif isinstance(managed, list):
                    for item in managed:
                        if getattr(item, "name", "") == "long_writer_agent":
                            agent = item
                            break

                if agent is None:
                    raise RuntimeError("create_agent 返回对象中未找到名为 long_writer_agent 的 managed agent")

            _normalize_agent_output_paths(agent, init_cwd=init_cwd, target_cwd=previous_cwd)
        except Exception as e:
            raise RuntimeError(
                "默认 CLI agent 构建失败（已切换为正式运行时同款）。"
                "请检查环境变量与依赖，或通过 agent_config.factory 指定自定义工厂。"
                f" 原因: {e}"
            ) from e
        finally:
            os.chdir(str(previous_cwd))

    overrides = config.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("`agent_config.overrides` 必须是 JSON object")
    for key, value in overrides.items():
        setattr(agent, key, value)

    _patch_agent_logger_for_console(agent)

    return agent


def _json_or_plain(value: str) -> Any:
    """
    主要作用：优先按 JSON 解析输入，失败时退回普通文本。

    输入参数：
    - value (str): 待格式化或待输出的任意值。

    返回值：
    - Any：返回该方法的主要输出结果。

    实现逻辑：
    - 读取方法所需输入并做必要预处理。
    - 执行该方法对应的核心业务逻辑。
    - 返回结果或通过副作用更新状态、日志和文件。
    """
    text = str(value).strip()
    if text == "":
        return ""
    try:
        return json.loads(text)
    except Exception:
        return value


def _assign_nested(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """
    主要作用：把值按点路径写入嵌套字典。

    输入参数：
    - target (Dict[str, Any]): 待写入值的目标字典。
    - dotted_key (str): 以点号分隔的嵌套字段路径。
    - value (Any): 待格式化或待输出的任意值。

    返回值：
    - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

    实现逻辑：
    - 读取方法所需输入并做必要预处理。
    - 执行该方法对应的核心业务逻辑。
    - 返回结果或通过副作用更新状态、日志和文件。
    """
    parts = [p for p in dotted_key.split(".") if p]
    if not parts:
        raise ValueError("空 key 不合法")

    cursor: Dict[str, Any] = target
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"路径冲突: {dotted_key}")
        cursor = child
    cursor[parts[-1]] = value


def _parse_key_value_pairs(pairs: List[str], field_name: str) -> Dict[str, Any]:
    """
    主要作用：将命令行键值对解析为结构化字典。

    输入参数：
    - pairs (List[str]): 命令行键值对列表。
    - field_name (str): 字段名或提示字段标签。

    返回值：
    - Dict[str, Any]：返回结构化字典结果，便于后续工作流阶段继续消费。

    实现逻辑：
    - 扫描输入中的关键模式、字段或噪声片段。
    - 完成清洗、规范化、回填或结构化整理。
    - 返回更稳定、可复用的中间结果。
    """
    parsed: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"`{field_name}` 参数必须是 key=value 形式: {pair}")
        key, raw_value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"`{field_name}` 中存在空 key")
        _assign_nested(parsed, key, _json_or_plain(raw_value.strip()))
    return parsed


def _coerce_text_for_console(text: Any) -> str:
    """
    主要作用：把任意对象转成适合控制台展示的文本。

    输入参数：
    - text (Any): 待解析、清洗或重写的原始文本内容。

    返回值：
    - str：返回处理后的文本、提示词、章节内容或格式化字符串。

    实现逻辑：
    - 读取方法所需输入并做必要预处理。
    - 执行该方法对应的核心业务逻辑。
    - 返回结果或通过副作用更新状态、日志和文件。
    """
    content = str(text)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        content.encode(encoding)
        return content
    except Exception:
        return content.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _patch_agent_logger_for_console(agent) -> None:
    """
    主要作用：在组件 CLI 场景下改造日志器输出。

    输入参数：
    - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。

    返回值：
    - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

    实现逻辑：
    - 整理当前阶段的核心信息。
    - 按照既定格式写入日志、文件或状态对象。
    - 保证运行过程可回溯、可排障、可复现。
    """
    logger = getattr(agent, "logger", None)
    if logger is None:
        return
    original_log = getattr(logger, "log", None)
    if not callable(original_log):
        return

    def _safe_log(message, *args, **kwargs):
        """
        主要作用：包装日志调用，避免控制台编码异常中断。

        输入参数：
        - message: 待解析的输入消息文本，通常来自组件 CLI、模型输出或结构化协议消息。
        - *args: 该参数用于承载 `args` 相关的业务上下文或控制信息。
        - **kwargs: 额外初始化配置，会透传给父类代理或底层构造流程。

        返回值：
        - None：该方法主要通过更新对象状态、写文件、记录日志或调用外部服务产生副作用。

        实现逻辑：
        - 整理当前阶段的核心信息。
        - 按照既定格式写入日志、文件或状态对象。
        - 保证运行过程可回溯、可排障、可复现。
        """

        return original_log(_coerce_text_for_console(message), *args, **kwargs)

    logger.log = _safe_log


def _format_log_value(value: Any) -> str:
    """
    主要作用：将日志值格式化为单行可读文本。

    输入参数：
    - value (Any): 待格式化或待输出的任意值。

    返回值：
    - str：返回处理后的文本、提示词、章节内容或格式化字符串。

    实现逻辑：
    - 整理当前阶段的核心信息。
    - 按照既定格式写入日志、文件或状态对象。
    - 保证运行过程可回溯、可排障、可复现。
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            return value

    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _to_relative_path(path_value: str) -> str:
    """
    主要作用：将绝对路径转成相对路径。

    输入参数：
    - path_value (str): 待转换为相对路径的路径字符串。

    返回值：
    - str：返回处理后的文本、提示词、章节内容或格式化字符串。

    实现逻辑：
    - 读取方法所需输入并做必要预处理。
    - 执行该方法对应的核心业务逻辑。
    - 返回结果或通过副作用更新状态、日志和文件。
    """
    try:
        p = Path(str(path_value))
        if not p.is_absolute():
            return str(p)
        return str(p.relative_to(Path.cwd()))
    except Exception:
        try:
            return os.path.relpath(str(path_value), start=str(Path.cwd()))
        except Exception:
            return str(path_value)


def _get_component_cli_log_file(agent=None, run_id: Optional[str] = None, component_name: str = "component") -> str:
    """
    主要作用：计算组件 CLI 调试日志的保存路径。

    输入参数：
    - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
    - run_id (Optional[str]): 组件 CLI 调试运行的唯一标识。
    - component_name (str): 工作流组件名称。

    返回值：
    - str：返回处理后的文本、提示词、章节内容或格式化字符串。

    实现逻辑：
    - 整理当前阶段的核心信息。
    - 按照既定格式写入日志、文件或状态对象。
    - 保证运行过程可回溯、可排障、可复现。
    """
    if agent is not None and not run_id:
        existing = getattr(agent, "_component_cli_test_log_file", "")
        if existing:
            return _to_relative_path(str(existing))

    output_dir = getattr(agent, "_output_dir", "") if agent is not None else ""
    run_suffix = run_id or time.strftime("%Y%m%d_%H%M%S")
    safe_component = str(component_name or "component").replace(" ", "_")
    file_name = f"component_cli_test_log_{safe_component}_{run_suffix}.txt"

    if output_dir:
        log_path = str(Path(output_dir) / file_name)
    else:
        log_path = file_name

    log_path = _to_relative_path(log_path)

    if agent is not None:
        setattr(agent, "_component_cli_test_log_file", log_path)
    return log_path


def _write_component_cli_test_log(
    agent,
    component: JsonWorkflowComponent,
    input_payload: Dict[str, Any],
    output: Optional[Any] = None,
    error: Optional[Exception] = None,
    run_id: Optional[str] = None,
) -> str:
    """
    主要作用：将组件 CLI 的输入、输出和日志统一落盘。

    输入参数：
    - agent: 当前 LongWriterAgent 或兼容宿主对象，负责模型调用、工具调度、状态管理和日志写入。
    - component (JsonWorkflowComponent): 工作流组件实例，用于获取契约或执行组件逻辑。
    - input_payload (Dict[str, Any]): 该参数用于承载 `input_payload` 相关的业务上下文或控制信息。
    - output (Optional[Any]): 模型、技能或组件生成的输出文本。
    - error (Optional[Exception]): 异常信息或错误文本。
    - run_id (Optional[str]): 组件 CLI 调试运行的唯一标识。

    返回值：
    - str：返回处理后的文本、提示词、章节内容或格式化字符串。

    实现逻辑：
    - 整理当前阶段的核心信息。
    - 按照既定格式写入日志、文件或状态对象。
    - 保证运行过程可回溯、可排障、可复现。
    """
    log_path = _get_component_cli_log_file(
        agent,
        run_id=run_id,
        component_name=getattr(component, "name", "component"),
    )
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"[component-cli] timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"[component-cli] component: {getattr(component, 'name', 'unknown')}\n")
        f.write(f"[component-cli] argv: {json.dumps(sys.argv[1:], ensure_ascii=False)}\n")
        f.write("[component-cli] input:\n")
        f.write(_format_log_value(input_payload) + "\n")
        if error is None:
            f.write("[component-cli] output:\n")
            f.write(_format_log_value(output) + "\n")
            f.write("[component-cli] status: success\n")
        else:
            f.write("[component-cli] error:\n")
            f.write(_format_log_value(str(error)) + "\n")
            f.write("[component-cli] status: failed\n")
        f.write("=" * 100 + "\n")

    return log_path


def _default_value_from_schema_hint(hint: Any, key: str = "") -> Any:
    """
    主要作用：根据 schema 提示构造默认测试值。

    输入参数：
    - hint (Any): 该参数用于承载 `hint` 相关的业务上下文或控制信息。
    - key (str): 该参数用于承载 `key` 相关的业务上下文或控制信息。

    返回值：
    - Any：返回该方法的主要输出结果。

    实现逻辑：
    - 读取方法所需输入并做必要预处理。
    - 执行该方法对应的核心业务逻辑。
    - 返回结果或通过副作用更新状态、日志和文件。
    """
    text = str(hint or "").lower()

    if "str" in text:
        if key == "task":
            return "调试任务：请分析大模型前沿进展"
        if key == "conclusion_text":
            return "这是用于调试的结论文本。"
        return ""
    if "list" in text:
        return []
    if "dict" in text:
        return {}
    if "bool" in text:
        return False
    if "int" in text or "float" in text or "number" in text:
        return 0
    return ""


def _build_default_input_payload(component: JsonWorkflowComponent) -> Dict[str, Any]:
    """
    主要作用：为组件自动构造默认输入载荷。

    输入参数：
    - component (JsonWorkflowComponent): 工作流组件实例，用于获取契约或执行组件逻辑。

    返回值：
    - Dict[str, Any]：返回结构化字典结果，便于后续工作流阶段继续消费。

    实现逻辑：
    - 读取当前上下文中的关键字段。
    - 按既定模板和业务规则拼装输入结构。
    - 返回下游阶段可直接消费的提示词、映射或载荷。
    """

    schema = component.input_format or {}
    if not isinstance(schema, dict):
        return {}

    defaults: Dict[str, Any] = {}
    for key, hint in schema.items():
        defaults[str(key)] = _default_value_from_schema_hint(hint, str(key))
    return defaults


def run_component_cli(component: JsonWorkflowComponent, argv: Optional[List[str]] = None) -> int:
    """
    主要作用：为工作流组件提供统一 CLI 调试入口。

    输入参数：
    - component (JsonWorkflowComponent): 工作流组件实例，用于获取契约或执行组件逻辑。
    - argv (Optional[List[str]]): 该参数用于承载 `argv` 相关的业务上下文或控制信息。

    返回值：
    - int：返回状态码、计数值或其他数值结果。

    实现逻辑：
    - 读取方法所需输入并做必要预处理。
    - 执行该方法对应的核心业务逻辑。
    - 返回结果或通过副作用更新状态、日志和文件。
    """
    run_id = time.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        prog=f"python {Path(sys.argv[0]).name}",
        description=f"组件 CLI 调试入口（component={component.name}）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("payload", nargs="?", help="JSON 字符串（兼容旧模式）")

    # 输入构建
    parser.add_argument("--input-json", help="输入 JSON 字符串（等价 payload）")
    parser.add_argument("--input-file", help="输入 JSON 文件路径")
    parser.add_argument("--set", action="append", default=[], help="覆盖输入字段，支持嵌套：a.b.c=value")

    # agent_config 构建
    parser.add_argument("--agent-config-json", help="agent_config JSON 字符串")
    parser.add_argument("--agent-config-file", help="agent_config JSON 文件路径")
    parser.add_argument("--agent-factory", help="agent 工厂，格式 package.module:callable")
    parser.add_argument("--factory-kw", action="append", default=[], help="工厂 kwargs：key=value")
    parser.add_argument("--override", action="append", default=[], help="构建后 agent 属性覆盖：key=value")

    # 调试输出
    parser.add_argument("--print-contract", action="store_true", help="打印组件 contract 后退出")
    parser.add_argument("--pretty", action="store_true", help="将输出 JSON 进行 pretty-print")
    parser.add_argument("--debug", action="store_true", help="打印解析后的输入与 agent_config")

    args = list(sys.argv[1:] if argv is None else argv)
    try:
        ns = parser.parse_args(args)
    except SystemExit:
        return 2

    if ns.print_contract:
        print(json.dumps(component.contract(), ensure_ascii=False, indent=2))
        return 0

    # 读取原始输入
    raw_payload: Optional[Any] = None
    if ns.input_file:
        try:
            raw_payload = json.loads(Path(ns.input_file).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"读取 --input-file 失败: {e}", file=sys.stderr)
            return 2
    elif ns.input_json:
        try:
            raw_payload = json.loads(ns.input_json)
        except Exception as e:
            print(f"解析 --input-json 失败: {e}", file=sys.stderr)
            return 2
    elif ns.payload:
        try:
            raw_payload = json.loads(ns.payload)
        except Exception as e:
            print(f"Invalid JSON argument: {e}", file=sys.stderr)
            return 2

    if raw_payload is None:
        input_payload = _build_default_input_payload(component)
        raw_payload = {"input": input_payload}
        print(
            "[debug] 未提供输入参数，已自动使用默认调试入参。"
            "建议后续通过 --input-json / --input-file / --set 覆盖。",
            file=sys.stderr,
        )

    if not isinstance(raw_payload, dict):
        print("JSON 顶层必须是 object", file=sys.stderr)
        return 2

    input_payload = raw_payload.get("input", raw_payload)
    if not isinstance(input_payload, dict):
        print("组件输入必须是 JSON object", file=sys.stderr)
        return 2

    try:
        set_values = _parse_key_value_pairs(ns.set, "--set")
        for key, value in set_values.items():
            input_payload[key] = value
    except Exception as e:
        print(f"解析 --set 失败: {e}", file=sys.stderr)
        return 2

    # 读取 agent_config
    agent_config: Dict[str, Any] = {}
    if isinstance(raw_payload.get("agent_config"), dict):
        agent_config.update(raw_payload.get("agent_config", {}))

    if ns.agent_config_file:
        try:
            from_file = json.loads(Path(ns.agent_config_file).read_text(encoding="utf-8"))
            if not isinstance(from_file, dict):
                raise ValueError("顶层必须是 object")
            agent_config.update(from_file)
        except Exception as e:
            print(f"读取 --agent-config-file 失败: {e}", file=sys.stderr)
            return 2

    if ns.agent_config_json:
        try:
            from_json = json.loads(ns.agent_config_json)
            if not isinstance(from_json, dict):
                raise ValueError("顶层必须是 object")
            agent_config.update(from_json)
        except Exception as e:
            print(f"解析 --agent-config-json 失败: {e}", file=sys.stderr)
            return 2

    if ns.agent_factory:
        agent_config["factory"] = ns.agent_factory

    if ns.factory_kw:
        try:
            factory_kwargs = _parse_key_value_pairs(ns.factory_kw, "--factory-kw")
            base_factory_kwargs = agent_config.get("factory_kwargs", {})
            if not isinstance(base_factory_kwargs, dict):
                print("agent_config.factory_kwargs 必须是 object", file=sys.stderr)
                return 2
            base_factory_kwargs.update(factory_kwargs)
            agent_config["factory_kwargs"] = base_factory_kwargs
        except Exception as e:
            print(f"解析 --factory-kw 失败: {e}", file=sys.stderr)
            return 2

    if ns.override:
        try:
            override_values = _parse_key_value_pairs(ns.override, "--override")
            base_overrides = agent_config.get("overrides", {})
            if not isinstance(base_overrides, dict):
                print("agent_config.overrides 必须是 object", file=sys.stderr)
                return 2
            base_overrides.update(override_values)
            agent_config["overrides"] = base_overrides
        except Exception as e:
            print(f"解析 --override 失败: {e}", file=sys.stderr)
            return 2

    if ns.debug:
        print("[debug] component:", component.name, file=sys.stderr)
        print("[debug] input_payload:", json.dumps(input_payload, ensure_ascii=False), file=sys.stderr)
        print("[debug] agent_config:", json.dumps(agent_config, ensure_ascii=False), file=sys.stderr)

    try:
        agent = build_agent_for_component_cli(agent_config)
        output = component.run(agent, input_payload)
        log_path = _write_component_cli_test_log(agent, component, input_payload, output=output, run_id=run_id)
        print(f"[debug] component test log: {_to_relative_path(log_path)}", file=sys.stderr)
        if isinstance(output, (dict, list)):
            print(json.dumps(output, ensure_ascii=False, indent=2 if ns.pretty else None))
        else:
            print(output)
        return 0
    except Exception as e:
        try:
            agent_for_log = locals().get("agent")
            log_path = _write_component_cli_test_log(agent_for_log, component, input_payload, error=e, run_id=run_id)
            print(f"[debug] component test log: {_to_relative_path(log_path)}", file=sys.stderr)
        except Exception:
            pass
        print(f"Component run failed: {e}", file=sys.stderr)
        return 1
