import json
import concurrent.futures
from typing import Dict
from smolagents import Tool
from utils.common_utils import safe_json_parse

class ExpertEvaluationTool(Tool):
    name = "expert_evaluation"
    description = "第四步：并行调用5位不同视角的虚拟专家对技术进行打分，只要赞成票 >=3 即标记 approved=True。"
    
    inputs = {
        "cluster_data": {"type": "any", "description": "包含内涵与趋势的聚类数据字典"},
        "domain": {"type": "string", "description": "所属领域"}
    }
    output_type = "any"

    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.experts = [
            ("专家A-学术界", "技术的理论创新性和突破性"),
            ("专家B-产业界", "技术的落地可行性和商业转化潜力"),
            ("专家C-专利局", "该技术是否具有足够的独创性，且未被广泛公知"),
            ("专家D-投资人", "该技术的市场规模潜力和爆发时间点"),
            ("专家E-政策专家", "该技术是否符合宏观产业政策导向")
        ]

    def _evaluate_single(self, name: str, focus: str, payload_str: str) -> Dict:
        prompt = (
            f"你是【{name}】，评审侧重点：{focus}。\n"
            "根据以下数据，给出纯JSON结果，包含 approve(bool) 和 reason(string, 一句话精炼理由)：\n"
            f"{payload_str}"
        )
        for attempt in range(2):
            try:
                res = self.model([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
                data = safe_json_parse(res.content, fallback_type=dict)
                if "approve" in data and "reason" in data:
                    return {"expert_name": name, "approve": bool(data["approve"]), "reason": str(data["reason"])}
            except Exception:
                pass
        return {"expert_name": name, "approve": False, "reason": "模型解析失败"}

    def forward(self, cluster_data: Dict, domain: str) -> Dict:
        term = cluster_data.get("technology_term", "Unknown")
        print(f"\n📥 [Expert Skill] 正在邀请 5 位专家对【{term}】进行并发评审...")
        
        # 构建给专家看的脱水数据，防止 Context 太长
        payload = {
            "domain": domain,
            "technology": cluster_data.get("technology_term"),
            "problem": cluster_data.get("technology_problem"),
            "method": cluster_data.get("technology_method"),
            "application": cluster_data.get("application_direction"),
            "trend": cluster_data.get("trend", {})
        }
        payload_str = json.dumps(payload, ensure_ascii=False)
        
        votes = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self._evaluate_single, n, f, payload_str) for n, f in self.experts]
            for future in futures:
                votes.append(future.result())
                
        approvals = sum(1 for v in votes if v["approve"])
        cluster_data["votes"] = votes
        cluster_data["approved"] = approvals >= 3
        
        status = "通过 ✅" if cluster_data["approved"] else "未通过 ❌"
        print(f"📤 [Expert Skill] 评审结果: {approvals}/5 票赞成 -> {status}")
        
        return cluster_data