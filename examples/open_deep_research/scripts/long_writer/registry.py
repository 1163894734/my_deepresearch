from __future__ import annotations

from typing import Dict

from .base_component import JsonWorkflowComponent
from .component_abstract import AbstractWritingComponent
from .component_body import BodyWritingComponent
from .component_citation_validation import CitationValidationComponent
from .component_conclusion import ConclusionWritingComponent
from .component_introduction import IntroductionWritingComponent
from .component_keyword_search import KeywordSearchExpansionComponent
from .component_outline import OutlineGenerationReflectionComponent
from .component_references import ReferencesWritingComponent


def build_workflow_components() -> Dict[str, JsonWorkflowComponent]:
    """构建工作流组件注册表。"""
    components = [
        KeywordSearchExpansionComponent(),
        OutlineGenerationReflectionComponent(),
        IntroductionWritingComponent(),
        BodyWritingComponent(),
        AbstractWritingComponent(),
        ConclusionWritingComponent(),
        CitationValidationComponent(),
        ReferencesWritingComponent(),
    ]
    return {component.name: component for component in components}
