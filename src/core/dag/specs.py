from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langchain.tools import StructuredTool
import logging
logger = logging.getLogger(__name__)

class StepSpec(BaseModel):
    id: str = Field(..., description="Unique node identifier")  # remove in the future, instead of id of the node, use the id of the whole plan instead
    tool_name: str = Field(..., description="Reference to a tool in the registry")
    description: str = Field(..., description="Human-readable explanation")
    inputs_from: List[str] = Field(default_factory=list, description="Upstream node IDs")
    input_key_map: Dict[str, str] = Field(
        default_factory=dict,
    )

class PlanSpec(BaseModel):
    # id: str = Field(..., description="Unique plan identifier")
    steps: List[StepSpec]
    # description: str = Field(..., description="Usage for this plan")

class ExecutionContext:
    NODE_STATUS = {
        "queued",     # scheduled but not yet ready
        "waiting",    # dependencies not yet satisfied
        "pending",    # ready to run
        "running",    # currently executing
        "succeeded",  # finished successfully
        "failed",     # errored
        "retrying",   # scheduled for a retry after failure
        "skipped",    # deliberately skipped
        "paused",     # halted awaiting external action
        "timeout",    # exceeded allowed runtime
        "cancelled",  # cancelled by user or system
        "aborted",    # aborted due to unrecoverable error
        "unknown",    # initial or undefined state
    }

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.status: Dict[str, str] = {}
        self.errors: Dict[str, Exception] = {}

    def set_status(self, node_id: str, status: str):
        """
        Set the status for a node. Raises ValueError if status is invalid.
        """
        
        if status not in self.NODE_STATUS:
            self.status[node_id] = "unknown"
            logger.warning(f"Invalid status '{status}' for node '{node_id}', defaulting to 'unknown'.")
        self.status[node_id] = status

    def store_result(self, node_id: str, result: StructuredTool):
        """
        Store the result of a successful node execution and mark it succeeded.
        """
        self.data[node_id] = result
        self.set_status(node_id, "succeeded")

    def store_error(self, node_id: str, error: Exception):
        """
        Store the error for a failed node execution and mark it failed.
        """
        self.errors[node_id] = error
        self.set_status(node_id, "failed")

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