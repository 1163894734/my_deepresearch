---
name: get_frontier_tech_sop
description: 当你接到【识别前沿技术】或【分析技术趋势】的任务时，必须首先调用此工具获取操作指南（不需要传入任何参数）。不要在没有阅读此指南的情况下自行猜测流程。
enabled: true
type: sop
---

# 【前沿技术识别标准作业程序 (SOP)】

你需要在一个 Python 沙盒环境中，严格按照以下步骤编写代码并调用工具：

1. 调用 `filter_documents(documents=raw_documents, domain=target_domain)` 得到 `kept_docs`。
2. 将 `kept_docs` 传给 `cluster_documents(kept_docs=kept_docs)` 获取技术簇列表。
3. 声明空列表 `approved_clusters = []`，然后使用 for 循环遍历所有的技术簇：
   - 必须先调用 `extract_narrative(cluster_data=..., domain=target_domain, all_domain_docs=kept_docs)` 提取内涵。
   - 然后把上一步返回的数据传给 `expert_evaluation(cluster_data=..., domain=target_domain)`。
   - 如果返回值的 `approved` 为 True，将其加入 `approved_clusters`。
4. 循环结束后，整理 `approved_clusters` 生成结构化的 Markdown 研报。
5. 【重要落盘】严禁使用 `open()` 或任何旧版保存工具！你必须连续调用两次 `save_file` 工具完成归档：
   - 第一次（存数据）：将 `approved_clusters` 列表保存为 JSON。调用参数：`content=approved_clusters, out_dir=run_dir, file_name="frontier_identification_report", out_type="json"`
   - 第二次（存研报）：将整理好的研报字符串保存为 Markdown。调用参数：`content=你的研报字符串, out_dir=run_dir, file_name="frontier_identification_report", out_type="md"`
6. 必须通过 `final_answer(你的研报字符串)` 提交结果。

注意：环境中已为你注入变量 `raw_documents`、`target_domain` 和 `run_dir`。

"在第 4 步整理 approved_clusters 生成研报时，请使用字典中返回的真实键值，如：technology_term（技术名称）、technology_problem（解决的痛点）、technology_method（核心方法）和 application_direction（应用方向）等来撰写具体描述。"

【格式红线】无论输入的参考资料是什么语言，你最终输出的段落必须、严格使用全中文（简体）撰写，专业术语可保留英文缩写（如 LLM, API）。