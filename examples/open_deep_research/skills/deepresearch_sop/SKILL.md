---
name: deep_research_director_sop
description: 当你接到【深度研究规划】任务时，必须首先调用此工具获取操作指南（不需传参）。
enabled: true
type: sop
---

# 【动态深度研究标准作业程序 (Dynamic Deep Research SOP)】

你是一个高级 AI 科研总管。系统已升级为“全局内存总线”与“带有条件准出的动态反思循环”架构。你必须严格按照以下阶段编写 Python 代码并调度，**严禁跳过或篡改逻辑**：
【注意】：`tool_get_var` 和 `tool_set_var` 必须在以下步骤的指导下使用，不得随意使用！

### 阶段零：领域标定 (Domain Calibration)
1. 调用下属智能体 `calibrator_agent(task=target_topic)`。
2. 使用 `tool_get_var(key="calibrator_agent_result")` 获取结果字典，赋值给 `domain_context`。
3. 初始化一个循环检索上下文变量：`search_context = str(domain_context)`。

---

### 第一阶段：动态反思检索循环 (Dynamic Retry Loop)
你必须在代码中编写一个最多执行 3 次的 `for` 循环（`for attempt in range(3):`）。在循环内部，严格按顺序执行以下 A 到 D 步：

**A. 觅食与检索 (Foraging & Search)**
1. 调用 `forager_agent(task=target_topic, additional_args={"context": search_context})`。
2. 使用 `tool_get_var(key="forager_agent_result")` 获取检索词列表 `search_queries`。
3. 调用工具 `tool_academic_search(search_queries=search_queries, max_results_per_query=8, engine="arxiv")`。（数据会自动双向追加到全局内存，无需接收其巨型返回值）。

**B. 提纯与聚类 (Map-Reduce)**
1. 使用 `raw_papers = tool_get_var(key="retrieved_papers")` 获取当前累积的文献库。
2. 使用代码 `compressed_papers = tool_insight_extractor(raw_papers=raw_papers)` 接收压缩后的文献列表。
3. 调用工具 `tool_semantic_cluster(compressed_papers=compressed_papers)` 进行语义分组，并使用 `json.dumps(..., ensure_ascii=False)` 转为字符串 `clustered_json`。

**C. 裁判评估 (Evaluation)**
1. 调用 `analyst_agent(task="请从数据中提取主题与瓶颈。并在最后单起一行严格输出 STATUS: PASS 或 STATUS: FAIL | MISSING: [需补充的关键词]。注意：你必须先使用 tool_set_var 将完整评估文本存入键名 'analyst_agent_result'，然后再调用 final_answer 结束。", additional_args={"raw_data": clustered_json})`。
2. 将结果赋值给文本变量 `analyst_eval_result`。

**D. 条件准出判断 (Conditional Break)**
1. 在 Python 代码中检查 `analyst_eval_result`。
2. 如果字符串包含 `"STATUS: PASS"`，打印日志 "文献储备达标"，并使用 `break` 跳出 `for` 循环。
3. 如果字符串包含 `"STATUS: FAIL"`，使用 Python 字符串操作（如 `split`）提取出 `MISSING:` 后面的关键词。将这些词追加到 `search_context` 中（如：`search_context += f" 必须重点补充: {missing_words}"`），并允许循环继续执行下一轮。

---

### 第二阶段：大纲因果结晶 (Outlining)
（注意：必须在 `for` 循环结束后，且在循环外部执行此阶段）
1. 调用 `outliner_agent(task="生成大纲", additional_args={"theme_analysis": analyst_eval_result, "raw_data": clustered_json})`。
2. 使用 `tool_get_var(key="outliner_agent_result")` 获取极简版大纲字典，并赋值给 `outline_data`。

---

### 第三阶段：基建下载与落盘归档 (Download & Output)
1. 调用工具 `tool_paper_downloader(outline_data=outline_data, save_dir=run_dir)` 静默下载 PDF 并更新全局主键库 `PAPER_DB`。
2. 调用工具 `save_file(content=outline_data, out_dir=run_dir, file_name="deep_research_outline", out_type="json")` 将大纲安全落盘。
3. 调用工具 `save_file(content=tool_get_var(key="PAPER_DB"), out_dir=run_dir, file_name="final_papers", out_type="json")` 将检索到的论文安全落盘。
4. 调用 `final_answer("深度研究大纲生成与反思循环已完成")` 宣告任务成功。

注意：环境中已为你注入变量 `target_topic` 和 `run_dir`,直接使用，请勿用`tool_get_var`获取这两个变量。严禁在代码中 `print` 巨型文献数组！