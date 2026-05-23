import os
import json
import time
import logging
from utils.agent_helper import setup_run_env
from smolagents import CodeAgent, CustomAgent
from scripts.skill_loader import load_skills_from_directory
from scripts.custom_tools import *
# 引入所有所需工具 (合并自 run_write_agent.py)
from scripts.custom_tools import *
import utils.common_utils as common_utils

# =========================
# 1. 基础配置与日志
# =========================

def main():
    run_dir, logger = setup_run_env("deep_research_review")
    os.makedirs(os.path.join(run_dir, "pdfs"), exist_ok=True)
    model = common_utils.ModelProvider.get_model()
    skills = load_skills_from_directory("skills", model=model)
    print(f"📁 通用深度研究专属目录已创建: {run_dir}")

    # =========================
    # 2. 实例化深度研究子智能体 (从 run_deepresearch.py 原封不动搬运)
    # =========================
    
    calibrator_agent = CustomAgent(
        model=model,
        tools=[AcademicSearchTool(),SetVariableTool(),GetVariableTool()], 
        name="calibrator_agent",
        description="用于在陌生领域快速建立认知锚点。必须传入 'task' 参数（宏观主题）。输入目标主题target_topic，输出字典格式的结果并存入 calibrator_agent_result 全局变量",
        instructions="""你是一个跨学科领域标定专家。面对一个陌生的领域，你需要快速建立认知体系。
        请严格按以下步骤执行：
        1. 调用 `tool_academic_search` 工具，使用检索词 "{task} comprehensive review OR state of the art survey" 获取最多 5 篇顶级文献，使用semanticscholar进行检索。
        2. 阅读摘要后，提取出该领域的三个核心元数据。
        3. 【强制红线】使用 `tool_set_var` 工具将提取出的元数据存入变量 `calibrator_agent_result`，以供总控和其他智能体调用。存入的必须是一个纯净的 Python 字典对象（dict），格式如下：
        {
            "core_metrics": ["核心评价指标1", "指标2"],
            "dominant_paradigms": ["当前主导流派1", "流派2"],
            "critical_bottleneck": "最大的物理/工程/商业限制"
            "domain_limiters": ["aerospace", "aviation"]  # <=== 新增：提取2-3个该领域专属的排他性英文限定词，用于防止跨领域检索污染
        }
        4. 不要废话！通过 final_answer 提交的第三步的变量。
        """
    )

    forager_agent = CustomAgent(
        model=model,
        tools=[SetVariableTool(),GetVariableTool(),ParseJsonTool()],
        name="forager_agent",
        description="基于领域锚点，将宏观主题拆解为四维检索词。必须传入 'task' (主题)。并将 forager_agent_result 存入全局变量",
        instructions="""
        首先使用context = tool_get_var(calibrator_agent_result)获取context。
        你是一个高级情报检索专家。请结合用户的原主题和传入的 context，生成 8 个正交的英文检索 Query。然后请严格覆盖以下四个通用维度（每个维度2个Query）：
        必须严格覆盖以下四个通用维度（每个维度2个Query）：
        1. 历史范式转移 (Paradigm Shifts)
        2. context 中提及的 dominant_paradigms 的最新突破
        3. context 中提及的 critical_bottleneck 的失效分析与妥协方案
        4. 试图颠覆 context 中 core_metrics 的前沿黑天鹅技术
        【防漂移强制红线】：为了防止检索引擎返回其他领域的无关高引论文，你生成的每一个 Query 都【必须】包含领域限定后缀！
        生成检索词时核心专有名词必须使用双引号包裹以进行精确匹配
        具体做法：从 context 中提取 `domain_limiters`，并在每个 Query 末尾加上 `AND (limiter1 OR limiter2)`。
        例如：`historical paradigm shifts in composite materials AND (aerospace OR aviation)`。
        使用``tool_set_var``工具将生成的 8 个 Query 存入变量 `forager_agent_result`，格式必须是一个纯正的 Python List[str] 对象.
        5. 不要废话！通过 final_answer 提交的第四步的变量。
        """
    )

    analyst_agent = CustomAgent(
        model=model,
        tools=[ParseJsonTool(),SetVariableTool(),GetVariableTool()],
        name="analyst_agent",
        description="用于分析聚类后的情报数据。必须传入 'task' 参数。输出分析结果存入 analyst_agent_result 全局变量",
        instructions=f"""你是一个严苛的文献质检员。当前研究的终极目标已经注入变量 `target_topic`。你将接收到由 UMAP+HDBSCAN 层次聚类生成的聚类树数据（包含多层级 items）。请为最顶层的每个大聚类写一段 100 字的核心洞察。要求如下：                                 
            用domain_context = tool_get_var(calibrator_agent_result)获取该领域的基准标定信息。                                                                       
            使用raw_data = tool_get_var(clustered_data)获取原始层次聚类数据，，你需要再用raw_data['clusters']获取簇列表。                                                                                              
            打印列表中的第一条数据并查看结构。
            请你遍历 raw_data 中的每个顶级 cluster 进行【相关度交叉验证】：                                                           
            对每一个顶级聚类簇提取核心洞察                                                                                
            最后单起一行严格输出 STATUS: PASS 或 STATUS: FAIL | MISSING: [需补充的关键词]。
            将结果使用 tool_set_var 存入'analyst_agent_result'，再调用 final_answer 结束。""",
        additional_authorized_imports=["json", "collections"]
    )

    outliner_agent = CustomAgent(
        model=model,
        tools=[ParseJsonTool(),SetVariableTool(),GetVariableTool()],
        name="outliner_agent",
        description="基于动态深度的聚类树生成嵌套大纲。必须传入 'task' 参数。输出大纲对象存入 outliner_agent_result",
        instructions="""你是一个顶级学术综述架构师兼 Python 数据处理专家。
        你接收到的 `raw_data` 是一棵【动态深度】的 JSON 嵌套聚类树。树的深度取决于文献总量。
        
        【大纲生成红线 - 必须严格遵守】：
        1. 递归嵌套：无论树有几层，你都需要将其映射为 JSON 大纲。所有的非叶子节点必须使用 `sections` 包含下一级节点。
        2. 叶子节点定义：当你在 `items` 列表中遇到直接包含 `paper_id` 的对象时，说明到达了最底层小节（叶子节点）。
        3. 绝不单篇成节：绝对不能把单篇论文变成一个独立的小节！你必须把同一个底层簇内的所有论文归为一个 `section`。
        4. 挂载 ID 与摘要：对于每个【叶子节点】，你需要在 Python 代码中提取该簇内所有论文的 `paper_id`，合并成一个列表挂载在 `paper_ids` 字段；同时基于前几篇论文的 title/insight 提取一段 80 字的核心论点，赋值给 `core_argument` 字段。
        5. 非叶子节点（如章、大节）只保留 `chapter_title` 和 `sections` 字段，绝不挂载 `paper_ids` 或 `core_argument`。

        【期望的最终 JSON 大纲结构示例】：
        [
            {
                "chapter_title": "1. 全局架构与方法论",
                "sections": [
                    {
                        "chapter_title": "1.1 硬件层的突破",
                        "core_argument": "本节重点探讨了 3D 封装与 TSV 技术带来的散热瓶颈与高带宽增益...",
                        "paper_ids": ["p1", "p5", "p9", "p12", "p30", "p35", "p44", "p56", "p68", "p77"]
                    },
                    {
                        "chapter_title": "1.2 软件协议栈演进",
                        "core_argument": "讨论编译器层面的软硬件协同设计以及神经网络量化带来的收益...",
                        "paper_ids": ["p2", "p8", "p15", "p22", "p29", "p31", "p49", "p60", "p71", "p88", "p91"]
                    }
                ]
            }
        ]
        
        请编写 Python 代码解析这棵树，动态生成符合上述要求的列表对象。
        最后使用 `tool_set_var("outliner_agent_result", outline)` 将列表存入全局变量，并调用 `final_answer("大纲生成完毕")` 宣告完成。""",
        additional_authorized_imports=["json", "collections"]
    )
    outliner_agent = CustomAgent(
        model=model,
        tools=[ParseJsonTool(),SetVariableTool(),GetVariableTool()],
        name="outliner_agent",
        description="基于动态深度的聚类树生成嵌套大纲。必须传入 'task' 参数。输出大纲对象存入 outliner_agent_result",
        instructions="""你是一个顶级学术综述架构师兼 Python 数据处理专家。
        你接收到的 `raw_data` 是一棵【动态深度】的 JSON 嵌套聚类树（通常包含：顶级大簇 -> 次级小簇 -> 论文）。
        
        【核心痛点】：由于文献多达数百篇，如果你在代码中直接手写组装所有的 paper_ids 会导致输出超长被截断；但如果你只写一个极简的 for 循环，章节标题又会变成毫无意义的占位符，失去学术价值。
        
        【终极解决方案（混合编程法） - 必须严格遵守】：
        在编写 Python 代码时，你必须采用“先思考建立字典，后循环映射数据”的策略：
        
        第一步：在 Python 代码的开头，基于你对 `raw_data` 真实内容的阅读和学术理解，**手写一个主题映射字典**。为每一个顶级大簇和次级小簇构思极具学术深度的 `title` 和 `core_argument`。
        【红线警告】：必须严格确保小节序号和大章序号级联绑定！例如第1章下的小节是1.1, 1.2；第2章下的小节必须是2.1, 2.2；第3章下的小节必须是3.1, 3.2，绝不能全部写成1.x！
        
        第二步：编写一个递归的 `for` 循环去解析 `raw_data`，在循环中通过索引（index）去映射你在第一步写好的主题字典，同时自动提取底层的 `paper_id` 列表。
        
        你输出的 Python 代码结构必须严格类似于以下范例：
        ```python
        import json
        
        # 1. 你大脑中构思的学术主题字典（请务必根据各章节的大章序号级联编写，绝不能全部机械模仿 1.x ！）
        theme_mapping = {
            "top_0": {"title": "1. XXXX的底层物理机制", "core_argument": ""},
            "top_0_sub_0": {"title": "1.1 XXXX的界面热阻研究", "core_argument": "探讨了..."},
            "top_0_sub_1": {"title": "1.2 XXXX的键合工艺突破", "core_argument": "分析了..."},
            "top_1": {"title": "2. XXXX的前沿系统级应用", "core_argument": ""},
            "top_1_sub_0": {"title": "2.1 XXXX在高性能计算中的应用", "core_argument": "探讨了..."},
            "top_1_sub_1": {"title": "2.2 XXXX在光子集成领域的扩展", "core_argument": "分析了..."},
            # ... 务必覆盖真实数据中所有的簇索引，并保证序号严格逐级递增绑定
        }
        
        # 2. 读取传入的全局变量
        raw_data_dict = json.loads(tool_get_var("clustered_data"))
        
        # 3. 编写动态组装逻辑
        outline = []
        for i, top_cluster in enumerate(raw_data_dict['clusters']):
            top_key = f"top_{i}"
            fallback_title = f"第{i+1}章 " + top_cluster.get('items', [{}])[0].get('items', [{}])[0].get('title', '未知主题')[:20]
            chapter_info = theme_mapping.get(top_key, {"title": fallback_title, "core_argument": ""})
            
            chapter_node = {
                "chapter_title": chapter_info["title"],
                "sections": []
            }
            
            for j, sub_cluster in enumerate(top_cluster.get('items', [])):
                sub_key = f"top_{i}_sub_{j}"
                
                # 🚀 核心修复点：通过 f"{i+1}.{j+1} " 将小节的兜底序号与当前大章索引进行彻底的动态化强制绑定，不再有写死的 1.
                first_item = sub_cluster.get('items', [{}])[0]
                fallback_text = first_item.get('core_breakthrough', first_item.get('insight', '未知核心突破'))[:30]
                sub_fallback = f"{i+1}.{j+1} " + fallback_text
                
                sub_info = theme_mapping.get(sub_key, {"title": sub_fallback, "core_argument": "综合分析本节文献核心规律。"})
                
                # 自动提取 paper_ids (绝不允许把单篇文献独立成节)
                pids = []
                for paper in sub_cluster.get('items', []):
                    if 'paper_id' in paper:
                        pids.append(paper['paper_id'])
                
                chapter_node['sections'].append({
                    "chapter_title": sub_info["title"],
                    "core_argument": sub_info["core_argument"],
                    "paper_ids": pids
                })
                
            outline.append(chapter_node)
            
        # 4. 落盘
        tool_set_var("outliner_agent_result", outline)
        ```
        
        【绝对严禁】：
        1. 绝不允许使用我示例中的假名字，必须根据你读取到的真实文献数据去起标题！
        2. 最后必须调用 `final_answer("大纲生成完毕")`。
        """,
        additional_authorized_imports=["json", "collections"]
    )

    tools = [AcademicSearchTool(), InsightExtractorTool(), SemanticClusterTool(), SaveFileTool(), ParseJsonTool(),SetVariableTool(),GetVariableTool(),LocalPaperInjectorTool()]  # 基础工具
    tools.extend(skills)

    director_agent = CustomAgent(
        name="director_agent",
        description="负责深度研究的总体规划与调度。必须严格按照 `deep_research_director_sop` 中的步骤调用。",
        tools=tools,
        managed_agents=[calibrator_agent, forager_agent, analyst_agent, outliner_agent],
        model=model,
        instructions="""你是一个科研总架构师。配备了4个下级专家智能体。
        每次执行任务前，务必先调用 `deep_research_director_sop` 工具，严格按照里面的 6 个阶段步骤调度。
        【架构注意】：系统已升级为全局内存总线。巨型数组（如文献库）已自动存在内存中。请多使用 `tool_get_var` 获取数据，严禁在代码中 `print` 或硬编码巨长的文献 JSON 字符串！""",
        additional_authorized_imports=["json", "time", "ast", "re", "os"]
    )

    # =========================
    # 3. 实例化长文写作子智能体 (从 run_write_agent.py 原封不动搬运)
    # =========================
    
    section_writer_agent = CustomAgent(
        model=model,
        tools=[], 
        name="section_writer_agent",
        description="学术长文主笔。",
        instructions="""你是一名世界级的科技领域首席研究员。
        【任务】：根据传入的标题、核心论点、上下文逻辑和 RAG 语料，撰写学术报告的一个小节。
        
        【核心写作红线 —— 彻底消除拼接感】：
        1. 拒绝机械罗列：绝对不允许出现“论文A指出...另外论文B发现...”这种记流水账的句式。你必须将多篇文献的信息**深度融合成一个有机的学术观点**。
           - ❌ 错误示范：“[REF:p1]提出了一种新型冷却技术。同时，[REF:p2]指出了成本过高的问题。”
           - ✅ 正确示范：“尽管新型冷却技术在热管理上展现出显著优势[REF:p1]，但其居高不下的制备成本仍是当前规模化应用的主要瓶颈[REF:p2]。”
        2. 建立文献对话感：运用高级学术转折词（如：然而、相较之下、进一步地、本质上），让不同文献的数据和观点产生碰撞与对比。
        3. 强制学术引用：必须在具体的数据、实验或观点句末使用 [REF:pX] 格式进行引用。
        4. 格式要求：你必须且只能使用 python 代码块来提交结果，即 `final_answer("你的正文字符串")`，严禁直接输出纯文本。
        """
    )

    rag_tool = UniversalRAGTool()
    assembler_tool = ReportAssemblerTool(writer_agent=section_writer_agent, rag_tool=rag_tool)
    writer_tools = [ParseJsonTool(), FlattenOutlineTool(), assembler_tool, SaveFileTool(),SetVariableTool(),GetVariableTool(), GenerateBibliographyTool(), LoadFileTool(), GenerateAbstractTool(), GenerateConclusionTool(), MdToWordTool()]
    writer_tools.extend(skills)

    writer_director_agent = CustomAgent(
        name="writer_director_agent",
        description="最高总管。负责解析大纲并调用底层的自动组装引擎。",
        tools=writer_tools,
        model=model,
        additional_authorized_imports=["json"],
        instructions="""你是一个长文写作总架构师 (CodeAgent)。
        请从long_writer_section_sop获取写作指南，并严格按照指南要求执行代码，完成学术报告的撰写。
        """
    )

    # =========================
    # 4. 实例化顶层主控智能体 (Review Master)
    # =========================
    # 它本身没有任何业务提示词修改，只有调度代码
    review_agent = CodeAgent(
        name="review_agent",
        description="顶级综述主控引擎。端到端协调 Research 和 Write 两个阶段。",
        tools=[],
        managed_agents=[director_agent, writer_director_agent],
        model=model,
        instructions="""
        请严格执行以下 Python 代码逻辑：
        1. 呼叫 `director_agent`，传入 task="请根据全局主题启动深度研究，并执行你的标准大纲生成SOP"，等待其完成。
        2. 呼叫 `writer_director_agent`，传入 task="大纲已就绪，请执行你的标准长文撰写、排版及导出SOP"，等待其完成。
        3. 运行结束后，使用 final_answer("全流程深度研究与综述撰写已成功完成！") 退出。
        
        【强制红线】：不要尝试去读取 json 文件，不要在代码里做状态传递 (state)，完全信任并放权给子智能体。
        """
    )

    # =========================
    # 5. 状态注入与任务启动
    # =========================
    target_topic = "三维异构集成"
    
    # 向所有需要环境信息的 Agent 注入一致的字典变量
    for agent in [director_agent, writer_director_agent, review_agent]:
        agent.state.update({"target_topic": target_topic, "run_dir": run_dir})

    try:
        logger.info("🎬 Review Agent 端到端全流程主引擎启动...")
        review_agent.run("请依照指令完成任务")
        print(f"\n🎉 规划与撰写全部完成！产出物已落盘至: {run_dir}")
    except Exception as e:
        logger.error(f"❌ 发生致命错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()