from langchain.tools import StructuredTool
from typing import Dict, List

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, StructuredTool] = {}

    def register(self, name: str, tool: StructuredTool):
        self._tools[name] = tool

    def get(self, name: str):
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def get_tool_summaries(self) -> List[Dict[str, str]]:
        """
        Step 1 & 2 Combined: Iterates through the tools and selects only the 
        'name' and 'description' fields, returning a clean list of dictionaries.
        """
        
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "metadata": tool.args_schema.model_json_schema
            }
            for tool in self._tools.values() 
        ]

def registry_tools(obj) -> ToolRegistry:
    """
    Inspect any object `obj`, find all attributes that are StructuredTool,
    and register them into a new ToolRegistry.
    """
    registry = ToolRegistry()
    for name in dir(obj):
        tool = getattr(obj, name)
        if isinstance(tool, StructuredTool):
            registry.register(name, tool)
    
    return registry