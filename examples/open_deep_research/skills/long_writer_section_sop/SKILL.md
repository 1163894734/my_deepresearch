---
name: get_section_sop
description: 当你需要撰写具体的报告小节时，调用此工具获取写作指南。
enabled: true
type: sop
---

# 【深度研究报告-单章节撰写 SOP (Evidence-Grounded Writer)】

  请在严格按照以下流程进行任务执行，不要做任何多余的修改和循环！

  # 1. 展平大纲并存入共享变量
  使用load_file(run_dir+"/final_papers.json", as_json=True)工具读取final_papers文件内容，并将读取到的内容使用tool_set_var存入全局变量"PAPER_DB"。
  flat_data = tool_flatten_outline(outline_data=tool_parse_json(text=outline_str))
  tool_set_var(key="flat_data", value=flat_data)
  
  # 2. 调用自动化装配引擎 (它在底层帮你完成了所有的循环、RAG和写作调度)
  full_report = tool_assemble_report(main_title=flat_data['main_title'], tasks=flat_data['tasks'])
  save_file(content=full_report, out_dir=run_dir, file_name="final_academic_report", out_type="md")
  
  
  # 3. 归档落盘
  formatted_content = tool_generate_bibliography(report_content=full_report, tasks_data=flat_data['tasks'])
  save_file(content=formatted_content, out_dir=run_dir, file_name="final_academic_report_finale", out_type="md")

  final_answer("全篇万字学术报告已完美撰写并保存！")
  ```
  【注意】
  环境中已经注入了变量 `outline_str`（大纲字符串）和 `run_dir`（当前运行目录），请直接使用，不要通过工具获取这两个变量的值。
  你需要做的就是按照上面提供的代码框架，调用工具完成写作任务。
  所有复杂的循环、RAG、调度逻辑都已经被封装在了 `tool_assemble_report` 这个自动化装配引擎中，你只需要正确传入参数即可。