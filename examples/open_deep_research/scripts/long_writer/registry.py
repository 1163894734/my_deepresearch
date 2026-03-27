from __future__ import annotations

from typing import Dict

from .base_component import JsonWorkflowComponent
from .component_abstract import AbstractWritingComponent
from .component_body import BodyWritingComponent
# from .component_citation_validation import CitationValidationComponent
from .component_conclusion import ConclusionWritingComponent
from .component_introduction import IntroductionWritingComponent
from .component_keyword_search import KeywordSearchExpansionComponent
from .component_outline import OutlineGenerationReflectionComponent
from .component_references import ReferencesWritingComponent
from .component_academic_search import AcademicSearchComponent

def build_workflow_components() -> Dict[str, JsonWorkflowComponent]:
    """
    主要作用：构建全部工作流组件的注册表。

    输入参数：
    - 无：该方法不接收显式业务参数。

    返回值：
    - Dict[str, JsonWorkflowComponent]：返回结构化字典结果，便于后续工作流阶段继续消费。

    实现逻辑：
    - 读取当前上下文中的关键字段。
    - 按既定模板和业务规则拼装输入结构。
    - 返回下游阶段可直接消费的提示词、映射或载荷。
    """
    components = [
        KeywordSearchExpansionComponent(),
        OutlineGenerationReflectionComponent(),
        IntroductionWritingComponent(),
        BodyWritingComponent(),
        AbstractWritingComponent(),
        ConclusionWritingComponent(),
        # CitationValidationComponent(),
        ReferencesWritingComponent(),
        AcademicSearchComponent(),
    ]
    return {component.name: component for component in components}
