import asyncio
import time
import os
from smolagents import OpenAIModel 
from utils.common_utils import ModelProvider

from .agent_context import PipelineContext
from .agent_planner import PlannerAgent
from .agent_writer import SectionWriterAgent
from .agent_finalizer import FinalizerAgent
from .core_workspace import ReportWorkspace

async def run_long_writer_pipeline(task_description: str, output_dir: str):
    print(f"🚀 [Pipeline] 启动长篇报告生成流水线 | 输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    
    # 初始化核心资源
    model = ModelProvider.get_model() 
    workspace = ReportWorkspace(output_dir)
    
    # 🔥 初始化全局 Context 工具类
    context = PipelineContext(model, workspace)
    
    # ==========================================
    # 阶段 1：全局规划 (Planner)
    # ==========================================
    print("🗺️ [阶段 1] Planner: 开始规划大纲与检索全局文献...")
    planner = PlannerAgent(context) 
    plan_result = await asyncio.to_thread(planner.generate_plan, task_description)
    
    global_outline = plan_result.get("outline", "")
    global_citations = plan_result.get("citations", {})
    raw_sections = plan_result.get("sections", [])
    
    workspace.save_json("outline_state.json", plan_result)
    workspace.safe_write("report_draft.md", f"# {task_description}\n\n## 大纲\n{global_outline}\n\n")

    # 🔥 新增：清洗章节数据（计算 Markdown 层级，并屏蔽不需要并发撰写的收尾章节）
    body_sections = []
    global_section_idx = 1
    
    for sec in raw_sections:
        title = str(sec.get("title", "")).strip()
        sec_type = sec.get("type", "")
        
        # 1. 过滤：摘要、参考文献等，交给 Finalizer 统一处理，不用派工人去写
        if sec_type in ["abstract", "references"] or "参考" in title or "文献" in title or title.lower() in ["references", "bibliography"]:
            continue
            
        # 2. 计算 Markdown 层级 (1 -> ##, 1.1 -> ###, 1.1.1 -> ####)
        level_str = str(sec.get("level", "1"))
        level_depth = len(level_str.split('.'))
        if level_depth <= 1:
            md_prefix = "##"
        elif level_depth == 2:
            md_prefix = "###"
        else:
            md_prefix = "####"
            
        # 3. 注入格式化后的标题和全局排序索引，供 Writer 和 Finalizer 使用
        sec["formatted_title"] = f"{md_prefix} {title}"
        sec["global_index"] = global_section_idx
        
        body_sections.append(sec)
        global_section_idx += 1

    if not body_sections:
        print("❌ 致命错误：Planner 未能生成任何有效的章节大纲。")
        return

    # ==========================================
    # 阶段 2：并发撰写 (Writers)
    # ==========================================
    print(f"✍️ [阶段 2] Writers: 启动 {len(body_sections)} 个并发撰写任务...")

    async def write_single_section(sec_data):
        writer = SectionWriterAgent(context)
        return await asyncio.to_thread(
            writer.write_section, 
            task=task_description, 
            global_outline=global_outline, 
            section=sec_data,
            base_citations=global_citations
        )

    completed_sections = await asyncio.gather(*(write_single_section(sec) for sec in body_sections))
    
    merged_citations = global_citations.copy()
    for sec_res in completed_sections:
        merged_citations.update(sec_res.get("new_citations", {}))
        
    workspace.save_json("completed_sections.json", {"sections": completed_sections, "citations": merged_citations})

    # ==========================================
    # 阶段 3：全局整合 (Finalizer)
    # ==========================================
    print("🎁 [阶段 3] Finalizer: 开始生成全局摘要与排版...")
    finalizer = FinalizerAgent(context)
    final_report = await asyncio.to_thread(
        finalizer.compile_report,
        task=task_description,
        global_outline=global_outline,
        compiled_sections=completed_sections,
        all_citations=merged_citations
    )
    
    workspace.safe_write("report_final.md", final_report, mode="w")
    
    cost_mins = (time.time() - start_time) / 60
    print(f"✅ [Pipeline] 全部完成！总耗时: {cost_mins:.2f} 分钟。最终文件: {output_dir}/report_final.md")

if __name__ == "__main__":
    task = "撰写一份关于具身智能前沿研究的深度综述"
    asyncio.run(run_long_writer_pipeline(task, "./outputs/pipeline_test"))