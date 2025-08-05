import networkx as nx
from typing import Dict, Any, List, Type
from .specs import PlanSpec, ExecutionContext, StepSpec
from langchain.tools import StructuredTool
from pydantic import BaseModel, create_model

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, StructuredTool] = {}

    def register(self, name: str, tool: StructuredTool):
        self._tools[name] = tool

    def get(self, name: str):
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def get_tool_summaries(self) -> List[Dict[str, str]]:
        """
        Step 1 & 2 Combined: Iterates through the tools and selects only the 
        'name' and 'description' fields, returning a clean list of dictionaries.
        """
        
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "metadata": tool.args_schema.model_json_schema
            }
            for tool in self._tools.values() 
        ]

def registry_tools(obj) -> ToolRegistry:
    """
    Inspect any object `obj`, find all attributes that are StructuredTool,
    and register them into a new ToolRegistry.
    """
    registry = ToolRegistry()
    for name in dir(obj):
        tool = getattr(obj, name)
        if isinstance(tool, StructuredTool):
            registry.register(name, tool)
    
    return registry

    
class DAGBuilder:
    def __init__(self, plan: PlanSpec):
        self.plan = plan

    def build(self) -> nx.DiGraph:
        dag = nx.DiGraph()
        for step in self.plan.steps:
            dag.add_node(step.id, spec=step)
        for step in self.plan.steps:
            for src in step.inputs_from:
                dag.add_edge(src, step.id)
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("The built graph is not a DAG.")
        return dag
    

class LLMDAGExecutor:
    def __init__(self, dag: nx.DiGraph, registry: ToolRegistry):
        self.dag = dag
        self.registry = registry

    def run(self, context: ExecutionContext ,**global_inputs):
        for node_id in nx.topological_sort(self.dag):
            step: StepSpec = self.dag.nodes[node_id]["spec"]
            tool = self.registry.get(step.tool_name)
            tool_input = self._prepare_input(step, context, global_inputs)
            context.set_status(node_id, "running")
            try:
                result = tool.invoke(tool_input)
                context.store_result(node_id, result)
                
            except Exception as e:
                context.store_error(node_id, e)
                raise

    def add_nodes(self, nodes: List[StepSpec]):
        for step in nodes :
            self.dag.add_node(step.id, spec=step)    

    def _prepare_input(
        self,
        step: StepSpec,
        context: ExecutionContext,
        global_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = global_inputs  # start with { "req": text, … }
        if "req" not in global_inputs:
            raise RuntimeError(f"Missing requirement for the agent")
        for arg_name, src in step.input_key_map.items():
            if src == "__GLOBAL__":
                merged[arg_name] = global_inputs[arg_name]
            else:
                if context.has_result(src):                
                    upstream_val = context.get_result(src)
                    merged[arg_name] = upstream_val        # even if upstream_val is None
                else:
                    raise RuntimeError(
                    f"Logic error: Result for dependency '{src}' not found in context "
                    f"when preparing input for step '{step.id}'."
                )
        return merged
    
# def _check_params():
