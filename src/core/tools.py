from typing import Any, Optional, TypeVar, Union, Type,Dict,List, Callable

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, ConfigDict, field_validator
import re
from helper import HelperRegistry

T = TypeVar("T", bound=BaseModel)

def _normalized_text(text: Any) -> str:
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def tool_factory(
                *,
                name: str = None,
                llm: Optional[BaseChatModel] = None,
                schema: Optional[Union[Type[T], dict]] = None,
                prompt_tpl: Optional[str] = None,
                description: Optional[str] = None,
                post_hook: Optional[Callable[[Any], Any]] = None,
                pre_hook: Optional[Callable[[Any, dict[str, Any]], Any]] = None,
                schema_id: Optional[str] = None,

                ) -> StructuredTool:
    
    _tpl = prompt_tpl or getattr(schema, "SYS_PROMPT", None)
    _helper_class = getattr(schema, "HELPER", None)
    class _InputModel(BaseModel):
        req: str = Field(..., description="User input string")
        model_config = ConfigDict(extra='allow')
        
        @field_validator("req", mode="before")
        def normalise_req(cls, v: Any) -> str:
            v = pre_hook(v) if pre_hook else v
            return _normalized_text(v)

    def _tool_helper(req: str, **kwargs) -> Any:
        if not llm and not post_hook:
            raise BrokenPipeError("Invalid setup")
        payload = {
            "req" : req,
            **kwargs
        }
        _helpers = HelperRegistry.get_helpers(schema_id=schema_id, list_classes=_helper_class)
        if llm:
            prompt = _tpl.format(**payload) if _tpl else req
            runnable = llm.with_structured_output(schema, method="tool_calling")
            output = runnable.invoke([HumanMessage(content=prompt)])
        else:
            output = payload
        output = post_hook(output, _helpers) if post_hook else output

        return output
    
    return StructuredTool.from_function(
        func=_tool_helper,
        name=name if name else f"{llm.__class__.__name__}",
        description=description
                    or f"LLM-powered tool built via tool_factory using {llm.__class__.__name__}",
        args_schema=_InputModel,
        return_direct=False,
    )
    
