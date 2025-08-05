import threading
from typing import Dict, Any, Type, List, Iterable


from typing import Callable, Any, List
from functools import partial


def _apply_helper_run(x: Any, helper: Any) -> Any:
    return helper.run(x)


def make_post_hooks(helpers: Dict[str, Any]) -> List[Callable[[Any], Any]]:
    """Wrap every helper's `run()` method into a post-hook."""
    return [partial(_apply_helper_run, helper=h) for h in helpers.values()]

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
    def get_helpers(cls, *, schema_id: str, list_classes) -> Dict[str, Any]:
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
        bucket = cls._cache.get(schema_id)
        if bucket is not None:
            return bucket

        # Slow path: build under lock
        with cls._lock:
            # Double-check pattern after acquiring lock
            bucket = cls._cache.get(schema_id)
            if bucket is None:
                cls._cache[schema_id] = {
                    helper.__name__: helper() for helper in list_classes
                }
            return cls._cache[schema_id]

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
