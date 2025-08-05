from typing import ClassVar, List, Union
from pydantic import BaseModel, Field
from core.dag import PlanSpec, ToolRegistry

class PlannerAgentFm(BaseModel):
    """
    Ask an LLM to pick which tools to run and their execution order.
    """
    SYS_PROMPT: ClassVar[str] = (
        """
        You are a planning assistant. Based on the user's FEATURES, generate a step-by-step execution plan 
        Your job: produce a JSON object that parses into PlanSpec (see JSON schema below).
        Rules:
        • Use ONLY the tools listed in LIST_TOOLS.
        • Every step id must be unique, lowercase, snake_case.
        • Define dependencies via inputs_from and input_key_map.
        • The very first step may read the user request via the special token "__GLOBAL__".
        • You only have to fill all primary key
        FEATURES: {req}\n
        LIST_TOOLS: {list_tools}
        """
    )
    steps: PlanSpec = Field(..., description="A plan with tool names and parameters.")
