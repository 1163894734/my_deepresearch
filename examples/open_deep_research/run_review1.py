import os
import json
import time
import logging
from enum import Enum
from dataclasses import dataclass
from utils.agent_helper import setup_run_env
from smolagents import CustomAgent, CodeAgent
from scripts.skill_loader import load_skills_from_directory
from scripts.custom_tools import *
import utils.common_utils as common_utils
from utils.state_manager import ResearchStateManager
from outline_builder import OutlineBuilder

def run_review1_core(target_topic="三维异构集成", run_timestamp="20260522_131108", resume_stage=None, run_dir=None, logger=None):
    # ==========================================
    # 1. 基础配置与环境初始化
    # ==========================================
    if run_dir is None or logger is None:
        run_dir, logger = setup_run_env("deep_review_pipeline_dag", run_timestamp=run_timestamp)
    else:
        os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "pdfs"), exist_ok=True)
    model = common_utils.ModelProvider.get_model()
    state = ResearchStateManager()
    
    resume_stage = (resume_stage or os.environ.get("REVIEW1_START_STAGE", "auto")).strip().lower()
    logger.info(f"📁 启动工业级 Python DAG 工作流 | 目标主题: {target_topic}")

    # ==========================================
    # 2. 实例化所有底层工具箱
    # ==========================================
    search_tool = AcademicSearchTool()
    insight_tool = InsightExtractorTool()
    cluster_tool = SemanticClusterTool()
    flatten_tool = FlattenOutlineTool()
    rag_tool = UniversalRAGTool()
    abs_tool = GenerateAbstractTool()
    conc_tool = GenerateConclusionTool()
    bib_tool = GenerateBibliographyTool()
    word_tool = MdToWordTool()
    set_var = SetVariableTool()
    get_var = GetVariableTool()
    local_injector = LocalPaperInjectorTool() 
    parse_json_tool = ParseJsonTool()
    save_file_tool = SaveFileTool(base_dir=run_dir)
    load_file_tool = LoadFileTool(base_dir=run_dir)

    # 预加载可能在内部隐式调用的skills
    skills = load_skills_from_directory("skills", model=model)

    # ==========================================
    # 3. 实例化子智能体 (完美复刻原版 Prompt)
    # ==========================================
    calibrator_agent = CustomAgent(
        model=model,
        tools=[search_tool, set_var, get_var, save_file_tool], 
        name="calibrator_agent",
        description="用于在陌生领域快速建立认知锚点。必须传入 'task' 参数（宏观主题）。输入目标主题target_topic，输出字典格式的结果并存入 calibrator_agent_result 全局变量",
        instructions="""你是一个跨学科领域标定专家。面对一个陌生的领域，你需要快速建立认知体系。
        请严格按以下步骤执行：
        1. 调用 `tool_academic_search` 工具，使用检索词 "{task} comprehensive review OR state of the art survey" 获取最多 5 篇顶级文献，使用semanticscholar进行检索。
        2. 阅读摘要后，提取出该领域的三个核心元数据。
        3. 【强制红线】使用 `save_file` 工具将提取出的元数据存入文件 `1_calibrator_agent_result.json`，以供总控和其他智能体调用。存入的必须是一个纯净的 Python 字典对象（dict），格式如下：
        {
            "core_metrics": ["核心评价指标1", "指标2"],
            "dominant_paradigms": ["当前主导流派1", "流派2"],
            "critical_bottleneck": "最大的物理/工程/商业限制",
            "domain_limiters": ["aerospace", "aviation"]  # <=== 新增：提取2-3个该领域专属的排他性英文限定词，用于防止跨领域检索污染
        }
        4. 不要废话！通过 final_answer 提交的第三步的变量。
        """
    )

    forager_agent = CustomAgent(
        model=model,
        tools=[set_var, get_var, parse_json_tool, load_file_tool, save_file_tool],
        name="forager_agent",
        description=f"基于领域锚点，将宏观主题拆解为四维检索词。必须传入 'task' (主题)。并将 forager_agent_result 存入全局变量",
        instructions="""
        首先使用context = load_file("1_calibrator_agent_result.json")获取context。
        你是一个高级情报检索专家。请结合用户的原主题和传入的 context，生成 8 个正交的英文检索 Query。
        必须严格覆盖以下四个通用维度（每个维度2个Query）：
        1. 历史范式转移 (Paradigm Shifts)
        2. context 中提及的 dominant_paradigms 的最新突破
        3. context 中提及的 critical_bottleneck 的失效分析与妥协方案
        4. 试图颠覆 context 中 core_metrics 的前沿黑天鹅技术
        【防漂移强制红线】：为了防止检索引擎返回其他领域的无关高引论文，你生成的每一个 Query 都【必须】包含领域限定后缀！
        生成检索词时核心专有名词必须使用双引号包裹以进行精确匹配
        具体做法：从 context 中提取 `domain_limiters`，并在每个 Query 末尾加上 `AND (limiter1 OR limiter2)`。
        例如：`historical paradigm shifts in composite materials AND (aerospace OR aviation)`。
        使用``save_file``工具将生成的 8 个 Query 存入文件 `2_forager_agent_result`，格式必须是一个纯正的 Python List[str] 对象.
        5. 不要废话！通过 final_answer 提交的第四步的变量。
        """
    )

    analyst_agent = CustomAgent(
        model=model,
        tools=[parse_json_tool, set_var, get_var, load_file_tool, save_file_tool],
        name="analyst_agent",
        description="用于分析聚类后的情报数据。必须传入 'task' 参数。输出分析结果存入 analyst_agent_result 全局变量",
        instructions="""你是一个严苛的文献质检员。当前研究的终极目标已经注入变量 `target_topic`。你将接收到由 UMAP+HDBSCAN 层次聚类生成的聚类树数据（包含多层级 items）。请为最顶层的每个大聚类写一段 100 字的核心洞察。要求如下：                                 
            用domain_context = load_file("1_calibrator_agent_result.json")获取该领域的基准标定信息。                                                                       
            使用raw_data = load_file("5_clustered_data.json")获取原始层次聚类数据，你需要再用raw_data['clusters']获取簇列表。                                                                                              
            打印列表中的第一条数据并查看结构。
            请你遍历 raw_data 中的每个顶级 cluster 进行【相关度交叉验证】：                                                           
            对每一个顶级聚类簇提取核心洞察                                                                                
            最后单起一行严格输出 STATUS: PASS 或 STATUS: FAIL | MISSING: [需补充的关键词]。
            将结果使用 save_file 工具存入'3_analyst_agent_result.json'，再调用 final_answer 结束。""",
        additional_authorized_imports=["json", "collections"]
    )

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
    
    assembler_tool = ReportAssemblerTool(writer_agent=section_writer_agent, rag_tool=rag_tool)

    # ==========================================
    # 全局状态注入
    # ==========================================
    for agent in [calibrator_agent, forager_agent, analyst_agent, section_writer_agent]:
        agent.state.update({"target_topic": target_topic, "run_dir": run_dir})

    # ==========================================
    # 4. 🚀 工业级硬编码工作流 (Python DAG + SaveFileTool)
    # ==========================================

    # 包装 SaveFileTool 的工具函数，确保所有落盘数据格式统一
    def persist_data(data, file_name, extension="json"):
        try:
            content_str = json.dumps(data, ensure_ascii=False, indent=2) if isinstance(data, (dict, list)) else str(data)
            save_file_tool.forward(content=content_str, out_dir=run_dir, file_name=file_name, out_type=extension)
            logger.info(f"  💾 中间产物已落盘 -> {file_name}.{extension}")
        except Exception as e:
            logger.warning(f"  ⚠️ 写入中间过程文件 {file_name}.{extension} 失败: {e}")

    class PipelineStage(str, Enum):
        PRELOAD = "preload"
        CALIBRATE = "calibrate"
        SEARCH = "search"
        CLUSTER = "cluster"
        OUTLINE = "outline"
        FLATTEN = "flatten"
        WRITE = "write"
        FINISH = "finish"

    @dataclass(frozen=True)
    class StageTransition:
        next_stage: str
        reason: str

    class ReviewPipelineStateMachine:
        def __init__(self):
            self.current_stage = self._normalize_stage(resume_stage)
            self.local_papers_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper_db")
            self.context = {
                "target_topic": target_topic,
                "search_context": f"用户原始主题: {target_topic}。",
                "max_retries": 3,
                "attempt": 0,
                "paper_source": "unknown",
                "cached_clustered_data": None,
                "cached_outline_data": None,
                "cached_flattened_tasks": None,
                "cached_full_report_draft": None,
                "cached_papers": None,
            }

        def _normalize_stage(self, stage_name: str) -> str:
            if stage_name in {"", "auto", "default"}:
                return PipelineStage.PRELOAD.value
            if stage_name == "clustered":
                return PipelineStage.CLUSTER.value
            if stage_name == "outline_data":
                return PipelineStage.OUTLINE.value
            if stage_name == "flattened":
                return PipelineStage.FLATTEN.value
            if stage_name == "draft":
                return PipelineStage.WRITE.value
            if stage_name in {item.value for item in PipelineStage}:
                return stage_name
            raise ValueError(f"不支持的起始阶段: {stage_name}。可选值: preload/calibrate/search/cluster/outline/flatten/write/finish/auto")

        def _transition(self, next_stage: str, reason: str):
            return StageTransition(next_stage=next_stage, reason=reason)

        def _load_run_dir_artifact(self, file_name: str, artifact_name: str):
            file_path = file_name
            if not os.path.exists(run_dir + "/" + file_path):
                raise ValueError(f"在 run_dir 中找不到 {artifact_name}: {file_path}")

            loaded_data = load_file_tool.forward(file_path=file_path, as_json=True)
            if isinstance(loaded_data, str):
                raise ValueError(f"run_dir 中的 {artifact_name} 无法解析为 JSON: {file_path}")

            logger.info(f"✅ 已加载 {artifact_name}: {file_path}")
            return loaded_data

        def build_outline_from_clusters(self, clustered_data):
            clusters = clustered_data.get("clusters", []) if isinstance(clustered_data, dict) else []
            outline = []

            for chapter_idx, top_cluster in enumerate(clusters, start=1):
                sub_clusters = top_cluster.get("items", []) if isinstance(top_cluster, dict) else []
                chapter_leaf_papers = self._collect_leaf_papers(top_cluster)
                chapter_titles = [paper.get("title", "") for paper in chapter_leaf_papers if paper.get("title")]
                chapter_phrase = self._derive_phrase_from_titles(chapter_titles, fallback=f"第{chapter_idx}章相关主题")
                chapter_node = {
                    "chapter_title": f"{chapter_idx}. {self._trim_text(chapter_phrase, 36)}",
                    "sections": []
                }

                for sub_idx, sub_cluster in enumerate(sub_clusters, start=1):
                    leaf_papers = self._collect_leaf_papers(sub_cluster)
                    section_no = f"{chapter_idx}.{sub_idx}"
                    section_title = self._derive_section_title(leaf_papers, section_no)
                    core_argument = self._derive_core_argument(leaf_papers, chapter_phrase)
                    paper_ids = [paper.get("paper_id") for paper in leaf_papers if paper.get("paper_id")]

                    chapter_node["sections"].append({
                        "chapter_title": section_title,
                        "core_argument": core_argument,
                        "paper_ids": paper_ids,
                    })

                outline.append(chapter_node)

            return outline

        def phase_preload(self):
            logger.info("\n▶️ [State] 本地文献库扫描与预加载...")
            if os.path.exists(self.local_papers_path):
                local_injector.forward(folder_path=self.local_papers_path)

            local_papers = state.get_all_papers()
            if local_papers:
                logger.info(f"✅ TinyDB 已加载本地文献，共 {len(local_papers)} 篇。")
                self.context["has_local_papers"] = True
                self.context["paper_source"] = "local"
                return self._transition(PipelineStage.CLUSTER.value, "本地文献已注入，直接进入提纯与聚类阶段")

            logger.info("ℹ️ TinyDB 暂无本地文献，将进入认知标定与动态检索流程。")
            self.context["has_local_papers"] = False
            self.context["paper_source"] = "remote"
            return self._transition(PipelineStage.CALIBRATE.value, "未发现本地文献，先做认知标定")

        def phase_calibrate(self):
            logger.info("\n▶️ [State] 启动认知标定...")
            calibrator_agent.run(f"目标主题: {target_topic}。请执行标定 SOP，注意替换 {{task}} 为目标主题。")
            calibrator_result = get_var.forward("calibrator_agent_result")
            persist_data(calibrator_result, "1_domain_calibration")
            self.context["calibrator_result"] = calibrator_result
            self.context["search_context"] = f"用户原始主题: {target_topic}。"
            return self._transition(PipelineStage.SEARCH.value, "认知标定完成，进入检索阶段")

        def phase_search(self):
            logger.info("\n▶️ [State] 启动检索状态...")
            if self.context["paper_source"] == "local":
                logger.info("  - 当前为本地文献模式，跳过检索状态，直接进入提纯与聚类。")
                return self._transition(PipelineStage.CLUSTER.value, "本地模式无需检索，直接进入提纯与聚类")

            if self.context["calibrator_result"] is None:
                calibrator_result = self._load_run_dir_artifact("1_domain_calibration.json", "calibrator_result")
                self.context["calibrator_result"] = calibrator_result
            attempt = self.context["attempt"]
            logger.info(f"  - 当前轮次: {attempt + 1}/{self.context['max_retries']}")

            logger.info("  - 启动四维检索策略生成...")
            forager_agent.run(f"{self.context['search_context']}。请结合领域 context，严格按照 4 个维度生成 8 个检索词。")
            search_queries = self._load_run_dir_artifact("2_forager_agent_result.json", "search_queries")

            if not search_queries or not isinstance(search_queries, list):
                logger.warning("  ⚠️ 检索词获取失败，启用基础兜底词组...")
                search_queries = [f"{target_topic} state of the art survey", f"{target_topic} critical bottlenecks", f"{target_topic} latest breakthrough"]

            persist_data(search_queries, f"2_search_queries_attempt_{attempt + 1}")

            logger.info("  - 启动全网文献并发爬取...")
            search_tool.forward(search_queries=search_queries, engine="semanticscholar")

            return self._transition(PipelineStage.CLUSTER.value, "检索完成，进入提纯与聚类阶段")

        def phase_cluster(self):
            logger.info("\n▶️ [State] 进入提纯与聚类阶段...")
            clustered_data = self.context["cached_clustered_data"]
            if clustered_data is None:
                raw_papers = state.get_all_papers()
                if not raw_papers:
                    raise ValueError("提纯与聚类阶段缺少可用文献，请先执行 preload/search，或先把文献注入 TinyDB。")

                logger.info(f"  - 当前可用文献数: {len(raw_papers)}，开始提纯...")
                compressed_papers = insight_tool.forward(raw_papers=raw_papers)
                persist_data(compressed_papers, "4_compressed_papers")

                logger.info("  - 开始语义聚类...")
                clustered_data = cluster_tool.forward(compressed_papers=compressed_papers)
                persist_data(clustered_data, "5_clustered_data")

            set_var.forward("clustered_data", json.dumps(clustered_data))
            self.context["cached_clustered_data"] = clustered_data

            if self.context["paper_source"] == "local":
                logger.info("  - 本地模式不做反向检索回跳，直接进入大纲阶段。")
                return self._transition(PipelineStage.OUTLINE.value, "本地文献已完成提纯与聚类，直接进入大纲")

            logger.info("  - 启动裁判评估与准出判断...")
            analyst_agent.run(f"研究终极目标为: {target_topic}。请读取 'clustered_data' 变量执行聚类交叉验证与质检。")
            analyst_result = str(get_var.forward("analyst_agent_result"))
            persist_data(analyst_result, f"5.5_analyst_evaluation_attempt_{self.context['attempt'] + 1}", extension="txt")

            if "STATUS: PASS" in analyst_result.upper():
                logger.info("✅ 裁判评估通过！文献储备已达标，进入大纲阶段。")
                return self._transition(PipelineStage.OUTLINE.value, "质检通过，进入大纲构思")

            if "STATUS: FAIL" in analyst_result.upper() or "MISSING:" in analyst_result.upper():
                missing_keywords = analyst_result.split("MISSING:")[-1].strip() if "MISSING:" in analyst_result.upper() else "相关前沿与瓶颈维度"
                logger.warning(f"⚠️ 裁判评估未达标，缺失维度: [{missing_keywords}]")

                if self.context["attempt"] < self.context["max_retries"] - 1:
                    self.context["attempt"] += 1
                    self.context["search_context"] += f" 【上一轮检索缺失，本轮必须重点补充定向检索：{missing_keywords}】"
                    logger.info("🔁 回跳到检索状态，继续补充文献。")
                    return self._transition(PipelineStage.SEARCH.value, f"质检未通过，缺失维度: {missing_keywords}，回跳检索")

                logger.warning("⚠️ 已达到最大反思检索次数 (3次)，强制进入下一阶段。")
                return self._transition(PipelineStage.OUTLINE.value, "达到最大检索轮次，强制进入大纲构思")

            logger.warning("⚠️ 裁判输出格式未严格遵循规范，默认视作文献足够，强制进入大纲生成。")
            return self._transition(PipelineStage.OUTLINE.value, "裁判输出格式异常，默认进入大纲构思")

        def phase_outline(self):
            logger.info("\n▶️ [State] 启动顶级大纲构思...")
            clustered_data = self.context["cached_clustered_data"]
            if clustered_data is None:
                clustered_data = self._load_run_dir_artifact("5_clustered_data.json", "clustered_data")
                self.context["cached_clustered_data"] = clustered_data

            if not clustered_data:
                raise ValueError("大纲阶段需要 clustered_data，但当前未能恢复。")
            outline_data = OutlineBuilder().build_outline_from_clusters(clustered_data)
            persist_data(outline_data, "6_outline_data")
            set_var.forward("outliner_agent_result", outline_data)
            self.context["cached_outline_data"] = outline_data
            return self._transition(PipelineStage.FLATTEN.value, "大纲生成完成，进入任务展平")

        def phase_flatten(self):
            logger.info("\n▶️ [State] 任务树展平与 PDF 预加载调度...")
            outline_data = self.context["cached_outline_data"]
            papers = self.context["cached_papers"]
            if outline_data is None:
                outline_data = self._load_run_dir_artifact("6_outline_data.json", "outline_data")
                self.context["cached_outline_data"] = outline_data
            if papers is None:
                papers = self._load_run_dir_artifact("4_compressed_papers.json", "compressed_papers")
                self.context["cached_papers"] = papers
            if not outline_data:
                raise ValueError("展平阶段需要 outline_data，但当前未能恢复。")

            flattened = flatten_tool.forward(outline_data, papers=papers, save_dir=os.path.join(run_dir, "pdfs"))
            persist_data(flattened, "7_flattened_tasks")
            self.context["cached_flattened_tasks"] = flattened

            tasks = flattened.get("tasks", [])
            if not tasks:
                raise ValueError("❌ 任务树展平失败，未能解析到任何有效章节任务，中断执行。")

            return self._transition(PipelineStage.WRITE.value, "任务展平完成，进入写作与导出")

        def phase_write(self):
            logger.info("\n▶️ [State] 进入核心组装引擎与收尾输出...")
            flattened = self.context["cached_flattened_tasks"]
            if flattened is None:
                flattened = self._load_run_dir_artifact("7_flattened_tasks.json", "flattened_tasks")
                self.context["cached_flattened_tasks"] = flattened
            if not flattened:
                raise ValueError("写作阶段需要 flattened_tasks，但当前未能恢复。")

            tasks = flattened.get("tasks", [])
            if not tasks:
                raise ValueError("写作阶段未找到有效任务列表。")

            logger.info(f"  - 共识别到 {len(tasks)} 个区块任务，开始组装长文...")
            full_report_draft = assembler_tool.forward(main_title=target_topic, tasks=tasks)
            persist_data(full_report_draft, "8_full_report_draft", extension="md")
            self.context["cached_full_report_draft"] = full_report_draft

            logger.info("  - 撰写前瞻摘要...")
            full_report = abs_tool.forward(full_report_draft, "请生成字数在 400 字左右，包含研究背景、核心方法对比、主要突破的高质量全文摘要。")
            logger.info("  - 撰写终章结论...")
            full_report = conc_tool.forward(full_report, "请生成字数在 500 字左右的结论，并对该领域未来的技术瓶颈和产业化趋势进行深度展望。")
            logger.info("  - 编排学术标准参考文献...")
            final_markdown = bib_tool.forward(full_report)

            logger.info("\n▶️ [State] 成果物理落盘导出...")
            save_file_tool.forward(content=final_markdown, out_dir=run_dir, file_name=f"综述_{target_topic}", out_type="md")

            try:
                word_tool.forward(md_content=final_markdown, out_dir=run_dir, file_name=f"综述_{target_topic}")
                logger.info("  - Word 版生成成功。")
            except Exception as e:
                logger.warning(f"  ⚠️ Word 转换跳过 (依赖或环境错误): {e}")

            logger.info(f"\n🎉 完美收官！全要素无阉割版 Python DAG 工作流执行完毕！产出物已保存至: {run_dir}")
            return self._transition(PipelineStage.FINISH.value, "写作与导出完成，流程结束")

        def run(self):
            stage_handlers = {
                PipelineStage.PRELOAD.value: self.phase_preload,
                PipelineStage.CALIBRATE.value: self.phase_calibrate,
                PipelineStage.SEARCH.value: self.phase_search,
                PipelineStage.CLUSTER.value: self.phase_cluster,
                PipelineStage.OUTLINE.value: self.phase_outline,
                PipelineStage.FLATTEN.value: self.phase_flatten,
                PipelineStage.WRITE.value: self.phase_write,
            }

            current_stage = self.current_stage
            while current_stage != PipelineStage.FINISH.value:
                handler = stage_handlers.get(current_stage)
                if handler is None:
                    raise ValueError(f"未知状态: {current_stage}")
                transition = handler()
                if isinstance(transition, StageTransition):
                    logger.info(f"↪️ 状态跳转: {current_stage} -> {transition.next_stage} | 原因: {transition.reason}")
                    current_stage = transition.next_stage
                else:
                    current_stage = transition

            return True

    try:
        ReviewPipelineStateMachine().run()
        return {"run_dir": run_dir, "success": True}

    except Exception as e:
        logger.error(f"\n❌ 工作流致命异常: {str(e)}", exc_info=True)
        raise


def main():
    run_review1_core()

if __name__ == "__main__":
    main()