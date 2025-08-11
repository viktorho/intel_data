from typing import Any, Optional, TypeVar, Union, Type,Dict,List, Callable, Tuple

from langchain.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, ConfigDict, field_validator
import re
from .helper import HelperRegistry, HookMergeManager
import logging
import asyncio, inspect
from functools import partial

logger = logging.getLogger(__name__)
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

def _payload_for_post(output, _helper=None):
    payload = {"_input": output}
    if _helper is not None:
        payload["_helper"] = _helper
    return payload

def _prep(schema, _tpl, llm, req: str, **kwargs) -> Tuple[Dict[str,Any], str, Any, bool]:
    payload = {"req": req, **kwargs}
    if not llm:
        return payload, "", None, False
    prompt   = _tpl.format(**payload) if _tpl else req
    runnable = llm.with_structured_output(schema, method="tool_calling")
    return payload, prompt, runnable, True

def _tool_usage_sync(schema, _tpl, llm, post_hook, _helper, req: str = "", **kwargs):
    payload, prompt, runnable, use_llm = _prep(schema, _tpl, llm, req, **kwargs)
    output = runnable.invoke([HumanMessage(content=prompt)]) if use_llm else schema(**payload)

    if post_hook:
        if inspect.iscoroutinefunction(post_hook):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                output = asyncio.run(post_hook(_input=output, _helper=_helper))
            else:
                fut = asyncio.run_coroutine_threadsafe(
                    post_hook(_input=output, _helper=_helper), loop
                )
                output = fut.result()
        else:
            payload = _payload_for_post(output, _helper)

            output = post_hook(**payload)

    return output



async def _tool_usage_async(schema, _tpl, llm, post_hook, _helper, req: str = "", **kwargs):
    payload, prompt, runnable, use_llm = _prep(schema, _tpl, llm, req, **kwargs)
    if use_llm:
        if hasattr(runnable, "ainvoke") and inspect.iscoroutinefunction(runnable.ainvoke):
            output = await runnable.ainvoke([HumanMessage(content=prompt)])
        else:
            output = await asyncio.to_thread(runnable.invoke, [HumanMessage(content=prompt)])
    else:
        output = schema(**payload)

    if post_hook:
        payload = _payload_for_post(output, _helper)

        if inspect.iscoroutinefunction(post_hook):
            output = await post_hook(**payload)
        else:
            output = await asyncio.to_thread(post_hook, **payload)

    return output


def tool_factory(
                *,
                name: str = None,
                llm: Optional[BaseChatModel] = None,
                schema: Optional[Union[Type[T], dict]] = None,
                prompt_tpl: Optional[str] = None,
                description: Optional[str] = None,
                post_hook: Optional[Callable[[Any, dict[str, Any]], Any]] = None,
                pre_hook: Optional[Callable[[Any, dict[str, Any]], Any]] = None,
                schema_id: Optional[str] = None,
                default_helper_kwargs: Dict[str, Dict] = None,
                ) -> StructuredTool:
    
    if not llm and not post_hook:
        raise BrokenPipeError("Invalid setup")
    
    _tpl = prompt_tpl or getattr(schema, "SYS_PROMPT", None)
    _helper_class = getattr(schema, "HELPER", None)

    _helper = None
    if _helper_class:
        _helper = HelperRegistry.get_helpers(schema_id=schema_id, 
                                            list_classes=_helper_class,
                                            helper_kwargs=default_helper_kwargs,
                                        )
        logger.info(f"Initializing {[h for h in _helper]} for {name}...")
    
    
    class _InputModel(BaseModel):
        req: str = Field(..., description="User input string")
        previous_id: List[str] = Field("__GLOBAL__", description="Get the id from the previous node")
        model_config = ConfigDict(extra='allow')
        
        @field_validator("req", mode="before")
        def normalise_req(cls, v: Any) -> str:

            v = pre_hook(_input=v) if pre_hook else v
            return _normalized_text(v)

    def _tool_helper(req: str="", **kwargs) -> Any:
        if not llm and not post_hook:
            raise BrokenPipeError("Invalid setup")
        
        payload = {
            "req" : req,
            **kwargs
        }

        if llm:
            prompt = _tpl.format(**payload) if _tpl else req
            runnable = llm.with_structured_output(schema, method="tool_calling")
            output = runnable.invoke([HumanMessage(content=prompt)])
        else:
            output = schema(**payload)

        if _helper:
            if post_hook is None:
                raise f"No function to provide {_helper}"
            
            return post_hook(_input=output, _helper=_helper)

        return post_hook(_input=output) if post_hook else output 
    
    return StructuredTool.from_function(
        func=_tool_helper,
        name=name if name else f"{llm.__class__.__name__}",
        description=description
                    or f"LLM-powered tool built via tool_factory using {llm.__class__.__name__}",
        args_schema=_InputModel,
        return_direct=False,
    )
    
