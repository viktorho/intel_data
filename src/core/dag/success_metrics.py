from __future__ import annotations
from datetime import date
from enum import Enum
from typing import List, Optional, Union, Any, ClassVar, Dict, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class DataSource(BaseModel):
    """Specifies the location and type of an input data source."""
    type: Literal["website", "api", "database", "local_file"]
    uri: str = Field(..., example="https://thuvienphapluat.vn")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Connection strings, credentials, headers, etc."
    )   

# ---------------------------------------------------------------------
# 2.  CORE METRIC (DATA-SOURCE ONLY, NO BENCHMARK YET)
# ---------------------------------------------------------------------


class Comparator(str, Enum):
    le = "<="
    lt = "<"
    ge = ">="
    gt = ">"
    eq = "=="

class MetricSpec(BaseModel):
    """
    Describes *what* will be measured and *how to fetch* the raw value.
    Benchmark fields (`comparator`, `target`) are optional and can be
    filled in later when you decide the pass/fail rule.
    """
    name: str = Field(..., example="coverage_rate")
    description: str = Field(
        ...,
        example="Percentage of legal documents successfully crawled "
                "versus official publication index."
    )

    unit: Optional[str] = Field(None, example="%")
    notes: Optional[str] = None

    comparator: Optional[Comparator] = None
    target: Optional[Union[int, float, str]] = None

    # ---------- Validation ----------
    @model_validator(mode="after")
    def _string_targets_must_use_eq(self) -> dict[str, Any]:
        t = self.target
        cmp = self.comparator
        if t is None and cmp is None:
            # perfectly fine – benchmark will be supplied later
            return self
        if (t is None) ^ (cmp is None):
            raise ValueError("Provide *both* comparator and target, or neither.")
        if isinstance(t, str) and cmp not in {Comparator.eq}:
            raise ValueError("String targets only valid with '==' comparator.")
        return self
    

class PromptPlFm(BaseModel):
    """
    Canonical representation of a project goal for the Planner agent.
    """

    description: str = Field(
        ...,
        min_length=15,
        example=(
            "Collect every legal document (2020-2024) from thuvienphapluat.vn, "
            "store as UTF-8 .txt grouped by Legislation Type, and generate a "
            "CSV containing full metadata."
        ),
    )

    success_metrics: List[MetricSpec] = Field(
        ..., min_items=1, description="At least one KPI **must** be defined."
    )

    priority: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Lower = higher priority (1 is top)."
    )

    @field_validator("description", mode="after")
    @classmethod
    def _trim_whitespace(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _unique_metric_names(cls, v):
        names = [m.name for m in v.success_metrics]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate metric names are not allowed.")
        return v


