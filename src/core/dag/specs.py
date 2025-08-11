from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain.tools import StructuredTool
import logging
from project_setup import log_value
import json, hashlib
logger = logging.getLogger(__name__)

class Goal(BaseModel):
    name: str
    target: float          
    op: str = ">=" 

class Guard(BaseModel):
    name: str
    limit: float 
    op: str = "<="

class StepSpec(BaseModel):
    id: str = Field(..., description="Unique node identifier")  # remove in the future, instead of id of the node, use the id of the whole plan instead
    tool_name: str = Field(..., description="Reference to a tool in the registry")
    description: str = Field(..., description="Human-readable explanation")
    inputs_from: List[str] = Field(default_factory=list, description="Upstream node IDs")
    input_key_map: Dict[str, str] = Field(
        default_factory=dict,
    )

class PlanSpec(BaseModel):
    plan_id: str = Field(..., description="Unique plan identifier")
    steps: List[StepSpec]
    goals: List[Goal] = Field(default_factory=list, description="Success conditions")
    guards: List[Guard] = Field(default_factory=list, description="Safety or quality constraints")
    description: str = Field(..., description="Usage for this plan")
    
    
    def evaluate_metrics(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Check whether the given metrics meet all goals and guards.
        Returns a dict with 'goals_passed' and 'guards_passed'.
        """
        def compare(value, threshold, op):
            if op == ">=": return value >= threshold
            if op == "<=": return value <= threshold
            if op == "==": return value == threshold
            raise ValueError(f"Unsupported op: {op}")

        goal_remain = [
            g.name
            for g in self.goals
            if not compare(metrics.get(g.name, float("-inf")), g.target, g.op)
        ]


        guard_remain = [
            gu.name
            for gu in self.guards
            if compare(metrics.get(gu.name, float("inf")), gu.limit, gu.op)
        ]

        return {
            "goal_remain": goal_remain,
            "guard_remain": guard_remain
        }

def _sha256(x: Any) -> str:
    try:
        s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, sort_keys=True)
    except Exception:
        s = repr(x)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

class ExecutionContext:
    # Node-level status only
    NODE_STATUS = {
        "queued",      # scheduled but not started
        "running",     # currently executing
        "succeeded",   # finished successfully
        "failed",      # errored
        "unknown",     # fallback
    }

    def __init__(self,
                 plan_id: str,
                 branch_id: Optional[str] = "",
                 run_id: Optional[str] ="",
                 plan_spec=None,               
                 ):
        
        self.plan_meta = {
            "plan_id": plan_id,
            "branch_id": branch_id,
            "run_id": run_id,
        }

        # storage
        self.data: Dict[str, Any] = {}
        self.status: Dict[str, str] = {}
        self.errors: Dict[str, Exception] = {}

        self.plan_spec = plan_spec            
        self.execution_sequence: List[str] = []  # logical order steps are STARTED

    def set_status(self, node_id: str, status: str):
        """
        Set the status for a node. Raises ValueError if status is invalid.
        """
        
        if status not in self.NODE_STATUS:
            self.status[node_id] = "unknown"
            logger.warning(f"Invalid status '{status}' for node '{node_id}', defaulting to 'unknown'.")
        self.status[node_id] = status

    # @log_value
    def store_result(self, node_id: str, result: StructuredTool):
        """
        Store the result of a successful node execution and mark it succeeded.
        """
        self.data[node_id] = result
        self.set_status(node_id, "succeeded")
        return node_id, result
    

    # @log_value
    def store_error(self, node_id: str, error: Exception):
        """
        Store the error for a failed node execution and mark it failed.
        """
        self.errors[node_id] = error
        self.set_status(node_id, "failed")
        return node_id, error
    
    def get_result(self, node_id: str) -> Any:

        """
        Retrieve the stored result for a node, or None if not present.
        """
        return self.data.get(node_id)
    
    def has_result(self, node_id: str) -> bool:
        """
        Check if a result exists for the given node ID.
        """
        return node_id in self.data
    
