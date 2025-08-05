from typing import List, ClassVar, Tuple,Dict, Any
import re
from pydantic import BaseModel, Field


class Feature(BaseModel):
    name: str
    dtype: str
    evidence: str
    is_primary_key: bool = Field(default=False, description="Whether this feature is a primary key")


class FeaturesFm(BaseModel):
    """
    Final list of features that will become table columns.
    """
    CORE_FIELDS: ClassVar[Tuple[str, ...]] = ("time_range", "location")

    features: List[Feature] = Field(..., description="List of generated features")

    SYS_PROMPT: ClassVar[str] = (
        "You are a data-analysis assistant.\n"
        "Given RequirementFm below, list as many useful features as possible "
        "for building a fact table.\n"
        "Return JSON like {{\"features\": [{{\"name\": ..., \"dtype\": ..., "
        "\"description\": ...}}, ...]}}.\n"
        "REQUIREMENTFM: {req}\n"
    )
    model_config = {"extra": "forbid"}

def feature_filter(feature: FeaturesFm):
    """
    """
    feature_ls = feature.features
    return [f.name for f in feature_ls]