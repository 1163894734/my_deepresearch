import os
import threading
from tinydb import TinyDB, Query
from tinydb.operations import set as db_set

class ResearchStateManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path="../outputs/research_state.json"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResearchStateManager, cls).__new__(cls)
                # 确保输出目录存在
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                if os.path.exists(db_path):
                    os.remove(db_path)
            
                cls._instance.db = TinyDB(db_path, encoding='utf-8')
                
                # 划分不同的业务表
                cls._instance.table_papers = cls._instance.db.table("papers")     # 替代 PAPER_DB 和 retrieved_papers
                cls._instance.table_reports = cls._instance.db.table("reports")   # 替代 report_list
                cls._instance.table_globals = cls._instance.db.table("globals")   # 替代游离的变量
                
        return cls._instance

    # ==================== 文献数据库操作 (PAPER_DB) ====================
    def upsert_paper(self, paper_data: dict):
        """插入或更新文献（去重核心）"""
        if not paper_data.get("id"):
            return
            
        with self._lock:
            PaperQuery = Query()
            # 根据 ID 查找，如果存在则更新，不存在则插入
            self.table_papers.upsert(paper_data, PaperQuery.id == paper_data["id"])

    def get_all_papers(self) -> list:
        """获取所有已检索的文献（替代 retrieved_papers 列表）"""
        with self._lock:
            return self.table_papers.all()

    def get_paper_by_id(self, paper_id: str) -> dict:
        with self._lock:
            PaperQuery = Query()
            result = self.table_papers.search(PaperQuery.id == paper_id)
            return result[0] if result else {}

    # ==================== 报告组装操作 (report_list) ====================
    def append_report_section(self, section_data: dict):
        with self._lock:
            self.table_reports.insert(section_data)

    def get_full_report(self) -> str:
        """按序号智能拼接生成完整报告（自动还原标题层级）"""
        with self._lock:
            sections = self.table_reports.all()
            # 按 idx 排序保证段落顺序
            sections.sort(key=lambda x: x.get("idx", 0))
            
            full_text = []
            for s in sections:
                content = s.get("content", "").strip()
                title = s.get("title", "")
                depth = s.get("depth")
                
                # 如果这条记录有明确的层级 depth 和标题，说明是正文结构，自动为它生成 Markdown 标题
                if title and depth is not None:
                    # 假设顶层是 H1，章节从 H2 开始，所以是 depth + 1
                    heading_markdown = "#" * (depth + 1)
                    full_text.append(f"{heading_markdown} {title}\n\n{content}")
                else:
                    # 针对 "摘要" 和 "结论" 这种直接在 custom_tools 里写死了 ## 格式的，或者没有 depth 的，直接拼接
                    full_text.append(content)
                    
            return "\n\n".join(full_text)

    # ==================== 通用全局变量 ====================
    def set_global_var(self, key: str, value):
        with self._lock:
            VarQuery = Query()
            self.table_globals.upsert({'key': key, 'value': value}, VarQuery.key == key)

    def get_global_var(self, key: str):
        with self._lock:
            VarQuery = Query()
            result = self.table_globals.search(VarQuery.key == key)
            return result[0]['value'] if result else None