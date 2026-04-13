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
注意：1. 代码执行环境中已为你注入了全局变量 `run_dir`，请直接使用它，严禁自行定义或赋值。
2. 当你使用 web_search 工具获取到搜索结果列表后，绝对不能仅仅基于搜索结果的简短 Snippet（摘要）就开始写解读报告！
你必须挑选出至少 2-3 个最相关的 URL，使用 `visit_page` 工具进去读取原文的详细技术原理。只有在阅读了全文后，才能提炼客观事实。
3. 将查到的资料用`print`打印出来,然后调用 `save_raw_materials_tool` 将完整的版本（不要总结， `web_search` 要将 `visited_page` 后的完整内容全部保存）保存在`run_dir`目录下的一个txt文件里，文件名可以是`{target_term}_raw_materials.txt`。

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
传入 `target_term`, `outline`, `facts`, `judged_drafts` 以及 `run_dir`。
注意：代码执行环境中已为你注入了全局变量 `run_dir`，请直接使用它，严禁自行定义或赋值。
最后使用 `final_answer()` 返回成功提示。