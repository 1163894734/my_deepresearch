# outline_builder.py
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import utils.common_utils as common_utils
import traceback

class OutlineBuilder:
    def __init__(self):
        self.model = common_utils.ModelProvider.get_model()
        self.max_workers = 8  # 可根据 API 并发限制调整

    @staticmethod
    def _trim_text(text: str, max_len: int = 36) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        return cleaned if len(cleaned) <= max_len else cleaned[:max_len].rstrip("-_/ ")

    @staticmethod
    def _collect_leaf_papers(node) -> list:
        leaf_papers = []

        def walk(current):
            if isinstance(current, dict):
                if current.get("paper_id"):
                    leaf_papers.append(current)
                    return
                for key in ("items", "chapters", "sections", "sub_chapters", "subsections", "level_2", "children"):
                    children = current.get(key)
                    if isinstance(children, list):
                        for child in children:
                            walk(child)
            elif isinstance(current, list):
                for child in current:
                    walk(child)

        walk(node)
        return leaf_papers

    def _call_llm(self, prompt: str) -> str:
        """
        封装 LLM 调用。
        兼容 smolagents 的模型接口（例如 LiteLLMModel、HfApiModel），
        它们通常通过 model(messages) 返回一个对象，再取 .content。
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.model(messages)
            # 根据实际返回类型提取文本
            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                # 尝试当成字典处理
                return response['content'].strip()
        except Exception as e:
            print(f"LLM call error: {e}")
            raise  # 让上层捕获

    def _generate_chapter_title(self, chapter_idx: int, leaf_papers: list) -> str:
        """为整个章节生成一个简短的英文标题（用于内部大纲）"""
        titles = [p.get("title", "") for p in leaf_papers if p.get("title")]
        titles_str = "\n".join(f"- {t}" for t in titles[:20])  # 限制数量
        prompt = f"""You are an academic editor. Based on the following paper titles, create a concise English chapter title (3-6 words) that captures the common theme.
Titles:
{titles_str}
Output ONLY the title, no extra text."""
        title = self._call_llm(prompt)
        # 清理可能的多余字符
        return self._trim_text(title.strip().strip('"'), 60)

    def _generate_section_info(self, section_no: str, leaf_papers: list) -> tuple:
        """为一个小节生成标题和论点，返回 (title, argument)"""
        paper_infos = []
        for p in leaf_papers:
            info = {
                "title": p.get("title", ""),
                "breakthrough": p.get("core_breakthrough") or p.get("insight") or "",
                "limitation": p.get("limitation") or ""
            }
            paper_infos.append(info)

        papers_json = json.dumps(paper_infos, ensure_ascii=False, indent=2)
        prompt = f"""You are an academic editor writing an outline for a review paper.
Below are the papers that belong to this subsection ({section_no}).
Create a concise English subsection title (3-6 words) and a Chinese core argument sentence (about 100 Chinese characters) summarizing the main breakthroughs and limitations.
Return ONLY a JSON object like:
{{"title": "English Title Here", "argument": "中文核心论点..."}}
Papers:
{papers_json}"""

        response = self._call_llm(prompt)
        try:
            result = json.loads(response)
            title = result.get("title", f"Section {section_no}")
            argument = result.get("argument", "内容待补充")
        except Exception:
            # 如果 JSON 解析失败，尝试提取文本（保守回退）
            title = f"Subsection {section_no}"
            argument = response[:200]  # 取前200字符作为粗略论点
        return title.strip(), argument.strip()

    def build_outline_from_clusters(self, clustered_data: dict) -> List[Dict[str, Any]]:
        clusters = clustered_data.get("clusters", []) if isinstance(clustered_data, dict) else []
        if not clusters:
            return []

        # 构建骨架和任务列表
        chapter_tasks = []
        section_tasks = []
        outline = []
        for ch_idx, top_cluster in enumerate(clusters, start=1):
            ch_leaf = self._collect_leaf_papers(top_cluster)
            chapter_tasks.append((ch_idx, ch_leaf))
            sections = []
            sub_clusters = top_cluster.get("items", []) if isinstance(top_cluster, dict) else []
            for sub_idx, sub in enumerate(sub_clusters, start=1):
                sub_leaf = self._collect_leaf_papers(sub)
                sec_no = f"{ch_idx}.{sub_idx}"
                section_tasks.append((ch_idx, sub_idx, sec_no, sub_leaf))
                sections.append({
                    "chapter_title": None,
                    "core_argument": None,
                    "paper_ids": [p.get("paper_id") for p in sub_leaf if p.get("paper_id")]
                })
            outline.append({
                "chapter_title": None,
                "sections": sections
            })

        # 并行执行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            # 提交章节标题生成
            for ch_idx, leaf in chapter_tasks:
                fut = executor.submit(self._generate_chapter_title, ch_idx, leaf)
                futures[fut] = ("chapter", ch_idx)
            # 提交小节信息生成
            for ch_idx, sub_idx, sec_no, leaf in section_tasks:
                fut = executor.submit(self._generate_section_info, sec_no, leaf)
                futures[fut] = ("section", ch_idx, sub_idx)

            for future in as_completed(futures):
                info = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Task failed: {info}, error: {e}\n{traceback.format_exc()}")
                    # 给失败的节点设置默认值
                    if info[0] == "chapter":
                        ch_idx = info[1]
                        outline[ch_idx - 1]["chapter_title"] = f"{ch_idx}. Untitled Chapter"
                    elif info[0] == "section":
                        ch_idx, sub_idx = info[1], info[2]
                        section = outline[ch_idx - 1]["sections"][sub_idx - 1]
                        section["chapter_title"] = f"Section {ch_idx}.{sub_idx}"
                        section["core_argument"] = "（自动生成失败，请手动补充）"
                    continue

                if info[0] == "chapter":
                    _, ch_idx = info
                    outline[ch_idx - 1]["chapter_title"] = f"{ch_idx}. {result}"
                elif info[0] == "section":
                    _, ch_idx, sub_idx = info
                    title, argument = result
                    section = outline[ch_idx - 1]["sections"][sub_idx - 1]
                    section["chapter_title"] = title
                    section["core_argument"] = argument

        return outline