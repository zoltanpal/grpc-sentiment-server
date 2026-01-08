from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

def to_dict(obj: Any) -> dict:
    """
    Normalize various Python objects into a dictionary.

    Conversion order:
      - Mapping -> dict(...)
      - dataclass -> asdict(...)
      - pydantic v2 -> .model_dump()
      - pydantic v1 -> .dict()
      - generic objects -> vars(obj)
      - None/unknown -> {}
    """
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return obj.model_dump()  # pydantic v2
        except Exception:  # pragma: no cover - best-effort
            pass
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()  # pydantic v1
        except Exception:  # pragma: no cover - best-effort
            pass
    if hasattr(obj, "__dict__"):
        try:
            return vars(obj)
        except Exception:  # pragma: no cover - best-effort
            pass
    return {}