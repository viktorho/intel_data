import threading
from typing import Dict, Any, List, Callable, Optional, Union
import asyncio

import inspect
from functools import partial
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, Future as CFut


def _apply_helper_run(x: Any, helper: Any) -> Any:
    return helper.run(x)


def make_post_hooks(helpers: Dict[str, Any]) -> List[Callable[[Any], Any]]:
    """Wrap every helper's `run()` method into a post-hook."""
    return [partial(_apply_helper_run, helper=h) for h in helpers.values()]


def inject_helper(
    keymap: Dict[str, str],
    *,
    required: Optional[set[str]] = None,
    remove_helper: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Create a decorator that maps items in `_helper` to function parameters.

    Args:
        keymap: map {func_param_name: helper_key}. Example:
                {"crawler": "WebCrawler", "lang_extractor": "BasicLangExtractor"}
        required: set of helper keys that must be present; raise if missing.
        remove_helper: if True, `_helper` is removed before calling the function.

    Usage:
        @DepInjector.from_helper({"crawler":"WebCrawler","lang_extractor":"BasicLangExtractor"})
        async def crawl_data(*, _input: CrawlerFm, crawler: WebCrawler, lang_extractor: BasicLangExtractor):
            ...
    """
    required = required or set(keymap.values())

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:

        @wraps(func)
        def wrapper(*args, **kwargs):
            helper: Dict[str, Any] = kwargs.get("_helper") or {}
            missing = [k for k in required if k not in helper]
            
            if missing:
                raise KeyError(f"Missing dependencies in _helper: {missing}")

            for param_name, helper_key in keymap.items():
                if param_name not in kwargs:
                    kwargs[param_name] = helper.get(helper_key)

            if remove_helper and "_helper" in kwargs:
                kwargs.pop("_helper")

            return func(*args, **kwargs)

        return wrapper

    return decorator

class HelperRegistry:
    """
    A thread-safe singleton cache for helper instances.

    • Each *tool* (or *schema*) gets its own bucket, indexed by `schema_id`.
    • Inside each bucket every helper class is instantiated exactly once.
    • The registry can be cleared per-schema or globally.

    Assumption
    ----------
    `schema_obj.HELPER` is an *iterable of classes* (List[Type] or Tuple[Type, …]).
    """

    # -------- class-level state --------
    _cache: Dict[str, Dict[str, Any]] = {}         # {schema_id: {ClassName: instance}}
    _lock: threading.Lock = threading.Lock()       # protects _cache (multi-thread / async)

    # ------------ public API -----------
    @classmethod
    def get_helpers(cls, *, schema_id: str,
                    list_classes:List[Any],
                    helper_kwargs: Optional[Dict[str, Dict]] = None,
                    ) -> Dict[str, Any]:
        """
        Return helper instances for a given schema.
        Creates and stores them lazily on first access.

        Parameters
        ----------
        schema_id   : unique key per tool (e.g. 'summarizer', 'translator_v2')
        schema_obj  : any object that exposes attribute `HELPER`

        Returns
        -------
        Dict[str, Any]   # {ClassName -> instance}
        """
        # Fast path: hit without lock (optimistic read)
        helper_kwargs = helper_kwargs or {}

        bucket = cls._cache.get(schema_id)
        if bucket is not None:
            return bucket

        # Slow path: build under lock
        with cls._lock:
            # Double-check pattern after acquiring lock
            bucket = cls._cache.get(schema_id)
            if bucket is None:
                cls._cache[schema_id] = {
                    helper.__name__: helper(**helper_kwargs.get(helper.__name__, {}))
                    for helper in list_classes
                }
            return cls._cache[schema_id]
        
    @classmethod
    def update_helpers(cls,
                       *,
                       schema_id:str,
                       helper_name:str,
                       **new_params)-> None:
        
        with cls._lock:
            bucket = cls._cache.get(schema_id)
            if bucket is None:
                raise KeyError(f"Schema 'schema_id' is empty") 
            helper = bucket.get(helper_name)
            if helper is None:
                raise KeyError(f"Helper '{helper_name}' do not contain in schema '{schema_id}.'") 
            if not hasattr(helper, "update_config"):
                raise AttributeError(
                    f"Helper {helper_name} have no update_config"
                ) 
            helper.update_config(**new_params)

    @classmethod
    def clear(cls, schema_id: str | None = None) -> None:
        """
        Remove cached helpers.

        • If `schema_id` is given → clear only that bucket.
        • Otherwise → clear the entire registry.
        """
        with cls._lock:
            if schema_id:
                cls._cache.pop(schema_id, None)
            else:
                cls._cache.clear()

    # ---------- optional helpers --------
    @classmethod
    def list_schemas(cls) -> List[str]:
        """Return all schema_ids currently cached."""
        return list(cls._cache.keys())

    @classmethod
    def cache_info(cls) -> Dict[str, int]:
        """Small introspection helper: how many helpers per schema."""
        return {sid: len(bucket) for sid, bucket in cls._cache.items()}