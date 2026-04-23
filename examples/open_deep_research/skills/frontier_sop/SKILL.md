---
name: frontier_tech_sensing_sop
description: 当你接到【前沿技术识别】、【弱信号挖掘】或【技术演进态势分析】任务时，必须首先调用此工具获取操作指南（不需传参）。严禁在未阅读此指南的情况下自行猜测流程。
enabled: true
type: sop
---

# 【全链路前沿技术遴选与态势感知标准作业程序 (Frontier Tech Sensing SOP)】

你是一个高级 AI 技术情报总管。系统已升级为“五步串联”数据流转架构。你必须在一个 Python 沙盒环境中，严格按照以下阶段编写 Python 代码并调度工具，**严禁跳过或篡改逻辑顺序**：

【注意】：系统已为你自动注入全局变量 `target_topic`（技术主题名，例如"脑机接口"）和 `run_dir`（工作目录）。严禁在代码中 `print` 巨型数据列表（如完整 JSON 数组）！

---

### 阶段零：多源数据采集 (Data Source & Collection)
1. 调用数据采集工具 `tool_multi_source_data_collector(domain=target_topic)`。
   *(注：该工具将跨越学术论文库、专利数据库、科技新闻、行业智库报告和项目数据等多源渠道进行定点采集)*
2. 获取返回的原始多源数据集，赋值给变量 `raw_multi_source_data`。

---

### 第一阶段：信息的结构化处理 (Information Structuring)
此阶段将碎片化、非结构化数据归一化为后续分析的标准底座。
1. 调用工具 `tool_fragmented_info_aggregator(multi_source_data=raw_multi_source_data, output_formats=["json", "excel", "markdown"], out_dir=run_dir, file_stem="01_structured_info")`。
2. 将工具返回的字典赋值给 `structured_data_result`。
3. 从结果中提取核心数据表：在代码中声明 `structured_table = structured_data_result["table"]`。

---

### 第二阶段：技术识别 (Technology Identification)
此阶段包含“关键前沿”与“弱信号”的双轨并行挖掘，必须依次执行：

**A. 关键前沿技术挖掘**
1. 调用工具 `tool_key_frontier_technology_mining(structured_table=structured_table, domain_keywords=[target_topic], output_format="dict", min_criteria=2, out_dir=run_dir, file_stem="02_key_frontier")`。
2. 将结果字典赋值给 `frontier_result`。此操作将自动抽取基本定义、解决痛点、核心原理、参数指标与作用价值。

**B. 弱信号技术识别**
1. 调用工具 `tool_weak_signal_technology_mining(structured_table=structured_table, domain_keywords=[target_topic], output_format="dict", min_criteria=2, out_dir=run_dir, file_stem="03_weak_signal")`。
2. 将结果字典赋值给 `weak_signal_result`。此操作将专门针对研究数量少、处于早期探索阶段的技术点进行聚类与抽取。

---

### 第三阶段：技术分析研判 (Analysis & Judgment)
此阶段基于上述成果，还原完整发展路径并研判未来态势：

**A. 演进路径分析**
1. 调用工具 `tool_evolution_path_analysis(frontier_list=frontier_result, weak_signal_list=weak_signal_result, structured_table=structured_table, output_format="dict", out_dir=run_dir, file_stem="04_evolution_path")`。
2. 将结果字典赋值给 `evolution_result`。*(该工具将自动提取主流技术路线、实现方案、难题瓶颈及优缺点)*。

**B. 重点技术演进态势感知**
1. 调用工具 `tool_key_technology_evolution_sensing(frontier_list=frontier_result, weak_signal_list=weak_signal_result, structured_table=structured_table, technology_name=target_topic, output_format="dict", out_dir=run_dir, file_stem="05_tech_sensing")`。
2. 将结果字典赋值给 `sensing_result`。*(该工具将评估技术的成熟度阶段，如早期探索/规模验证/工程化应用，并研判未来趋势)*。

---

### 第四阶段：成果输出与报告整合 (Result Output)
（注意：之前的工具执行时，若传入了 `out_dir`，文件已自动静默落盘。本阶段负责汇总并宣告任务完成）
1. 在 Python 代码中，从 `sensing_result` 提取两份核心专报的 Markdown 内容：
   ```python
   frontier_report = sensing_result.get("frontier_special_report", "")
   weak_signal_report = sensing_result.get("weak_signal_special_report", "")
   final_report = frontier_report + "\n\n---\n\n" + weak_signal_report