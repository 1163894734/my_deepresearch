---
name: interpretation_sop
description: 当你接到【技术名词解读】的任务时，必须首先调用此工具获取操作指南（不需要传入任何参数）。不要在没有阅读此指南的情况下自行猜测流程。
enabled: true
type: sop
---
# 🎯 技术名词解读 SOP (Interpretation Agentic Workflow)

当你接收到【技术名词解读】或【名词解释】的任务时，请**严格按照以下步骤**编写并执行 Python 代码。

## 🛠️ 标准操作流程

### 第一步：并行资料检索 (Retrieval)
1. 从用户输入中提取核心名词（`target_term`）。
2. 使用 `academic_search_tool` 和 `web_search` 获取该名词相关的技术原理和进展。
3. 将所有文本拼接成一个长字符串 `raw_materials`。

### 第二步：提炼事实与大纲 (Fact & Outline)
调用 `extract_fact_and_outline_tool` 工具，传入 `target_term` 和 `raw_materials`。
获取包含 `facts` (客观事实) 和 `outline` (动态大纲) 的字典。

### 第三步：多风格底稿并发撰写 (Drafting)
调用 `generate_all_drafts_tool` 工具，传入 `target_term`, `facts` 和 `outline`。
该工具将在底层自动并发生成"百科版"、"专报版"、"科普版"三版初稿，并返回包含这三版内容的字典 `drafts_dict`。

### 第四步：多风格并发审修 (Judging & Refining)
调用 `judge_all_drafts_tool` 工具，传入 `outline`, `facts` 和 `drafts_dict`。
该工具会在底层自动并发对三版初稿进行事实核查与修改，并返回定稿字典 `judged_drafts`。

### 第五步：组装与持久化 (Assembly & Save)
调用 `assemble_and_save_report_tool` 工具。
传入 `target_term`, `outline`, `facts`, `judged_drafts` 以及环境变量 `run_dir`。
最后使用 `final_answer()` 返回成功提示。