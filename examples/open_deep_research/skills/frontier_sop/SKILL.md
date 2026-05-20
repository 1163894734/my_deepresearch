---
name: frontier_tech_sensing_sop
description: 当你接到【前沿技术识别】、【弱信号挖掘】或【技术演进态势分析】任务时，必须首先调用此工具获取操作指南（不需传参）。严禁在未阅读此指南的情况下自行猜测流程。
enabled: true
type: sop
---

请按以下步骤顺序执行任务，每个步骤的输出需通过print打印，**严禁省略任何步骤**：
【注意】`run_dir` 变量已注入环境，指向当前运行目录，请直接使用，不要通过工具获取该变量的值，并且请不要写成"run_dir"这样的字符串。
0. 定义变量(必须执行)：
origin_paper_dir = "../paper_db"
paper_meta_dir = run_dir + "/papers"

1. 解析论文检索任务并批量检索保存：
调用工具 `tool_local_paper_injector(folder_path=origin_paper_dir)` 加载本地论文（此路径必须硬编码，绝不允许修改）。
调用工具 `paper_data = tool_get_var(key="retrieved_papers")` 获取注入后的结果数据。
调用工具 `save_file(content=paper_data, out_dir=paper_meta_dir, file_name="local_papers_injected", out_type="json")` 将动作B获取的数据保存。打印保存成功的信息。

2. 合并所有检索结果并标准化字段：
遍历`paper_meta_dir`目录中的所有文件，用load_file读取每个文件（结果为文献的元数据列表，参数as_json设置为True）。将所有文件的results列表拼接成一个总列表。然后调用heterogeneous_data_mapping工具对总列表进行字段转化（统一字段命名），最后将转化后的结果使用save_file工具保存到`run_dir`目录下，文件名为1_heterogeneous_data.json。打印保存成功信息。

3. 技术主题聚类分析：
从`run_dir`目录下读取1_heterogeneous_data.json文件（内容为列表，每个元素是论文元信息字典）。打印第一条数据并简要分析其结构。从每条数据中提取与内容高度相关的字段（至少包含：'标题'、'关键词'、'所属技术领域'）。将这些提取的字段作为参数调用technology_clusterer工具进行聚类。聚类结果中每个类别包含"cluster_label"字段。将聚类结果保存到`run_dir`目录下，文件名为2_heterogeneous_clusters.json。打印输出聚类簇的个数及每个簇的基本信息。

4. 技术信号评估：
从`run_dir`目录下读取2_heterogeneous_clusters.json文件（内容为列表，每个元素是一个聚类簇，簇包含papers字段记录该簇的论文列表）。调用tech_signal_evaluator工具对这些聚类簇进行技术信号评估（如成熟度、影响力、活跃度等）。将工具输出保存为3_technology_signals.json文件到`run_dir`目录。打印评估完成信息和关键指标摘要。

5. 技术内涵提取：
从`run_dir`目录下读取3_technology_signals.json文件。调用tech_connotation_extractor工具提取每个技术簇的技术内涵（如核心技术原理、关键使能技术、应用场景等）。将结果保存为4_technology_connotations.json文件到`run_dir`目录。打印提取完成信息。

6. 技术演进路径分析：
从`run_dir`目录下读取4_technology_connotations.json文件。调用tech_evolution_analyzer工具分析技术演进路径（如技术代际划分、关键里程碑、发展趋势等）。将结果保存为5_technology_evolution.json文件到`run_dir`目录。打印分析完成信息及演进路径概要。

7.总结前沿技术：
从`run_dir`目录下读取5_technology_evolution.json文件。调用frontier_report_generator工具撰写《关键前沿技术识别与分析报告》
使用save_file工具将《关键前沿技术识别与分析报告》保存为`run_dir`目录下的`key_frontier_technologies_report.md`
调用weak_signal_report_generator工具撰写《弱信号技术识别与分析报告》
使用save_file工具将《弱信号技术识别与分析报告》保存为`run_dir`目录下的`weak_signal_technologies_report.md`

8.转为word格式输出：
调用markdown_to_word工具将`key_frontier_technologies_report.md`转换为`key_frontier_technologies_report.docx`，并保存到`run_dir`目录下
调用markdown_to_word工具将`weak_signal_technologies_report.md`转换为`weak_signal_technologies_report.docx`，并保存到`run_dir`目录下
并使用final_answer返回简短的完成信号结束整个流程。