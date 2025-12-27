from typing import Optional

from agent.common.modeling import Model
from .object_detector import ObjectDetector, DetectObjects
from .searcher import Searcher, Search
from .text_extractor import TextExtractor, OCR
from .tool import Tool

TOOL_REGISTRY = [
    ObjectDetector,
    Searcher,
    TextExtractor,
]

ACTION_REGISTRY = {
    Search,
    DetectObjects,
    OCR,
}

IMAGE_ACTIONS = {
    OCR,
    DetectObjects,
}


def get_tool_by_name(name: str):
    for t in TOOL_REGISTRY:
        if t.name == name:
            return t
    raise ValueError(f'Tool with name "{name}" does not exist.')


def initialize_tools(config: dict[str, dict], llm: Optional[Model], device=None) -> list[Tool]:
    tools = []
    if config:
        for tool_name, kwargs in config.items():
            if kwargs is None:
                kwargs = {}
            kwargs.update({"llm": llm, "device": device})
            tool_class = get_tool_by_name(tool_name)
            t = tool_class(**kwargs)
            tools.append(t)
    return tools
