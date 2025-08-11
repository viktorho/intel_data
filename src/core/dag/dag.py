import networkx as nx
from typing import Dict, Any, List
from .specs import PlanSpec, ExecutionContext, StepSpec

import asyncio, inspect
from .tool_registry import ToolRegistry

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
    def __init__(self, dag: nx.DiGraph, registry: ToolRegistry, max_concurrency=None, fail_fast=True):
        self.dag = dag
        self.registry = registry
        self.max_concurrency = max_concurrency
        self.fail_fast = fail_fast


    async def run_async(self, context: ExecutionContext, **global_inputs):
        sem = asyncio.Semaphore(self.max_concurrency) if self.max_concurrency else None

        # Track how many dependencies each node still needs 
        dep_count = {n: sum(1 for _ in self.dag.predecessors(n)) for n in self.dag.nodes}
        ready = [n for n, count in dep_count.items() if count == 0]
        tasks = {}

        for node_id in ready:
            tasks[node_id] = asyncio.create_task(
                self._run_node(node_id, context, global_inputs, sem, dep_count, tasks)
            )
            context.set_status(node_id, "queued")

        pending = set(tasks.values())

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            # fail-fast: stop immediately on the first task that errored
            if self.fail_fast:
                for t in done:
                    exc = t.exception()
                    if exc is not None:
                        for p in pending:
                            p.cancel()
                        raise exc
            for t in tasks.values():
                if not t.done() and t not in pending:
                    pending.add(t)

        
    async def _run_node(self, node_id, context, global_inputs, sem, dep_count, tasks):
        if sem:
            async with sem:
                await self._execute_node(node_id, context, global_inputs, dep_count, tasks)
        else:
            await self._execute_node(node_id, context, global_inputs, dep_count, tasks)
    
    async def _execute_node(self, node_id, context, global_inputs, dep_count, tasks):
        step: StepSpec = self.dag.nodes[node_id]["spec"]
        tool = self.registry.get(step.tool_name)
        tool_input = self._prepare_input(step, context, global_inputs)
        context.set_status(node_id, "running")

        try:
            result = await self._invoke(tool, tool_input)
            context.store_result(node_id, result)

            # Schedule downstream nodes if all deps are done
            for succ in self.dag.successors(node_id):
                dep_count[succ] -= 1
                if dep_count[succ] == 0:
                    tasks[succ] = asyncio.create_task(
                        self._run_node(succ, context, global_inputs, sem=None, dep_count=dep_count, tasks=tasks)
                    )
                    context.set_status(succ, "queued")

        except Exception as e:
            context.store_error(node_id, e)
            if self.fail_fast:
                for t in tasks.values():
                    t.cancel()
                raise
    
    async def _invoke(self, tool, payload):
        """Handle sync or async tool invocation."""
        res = tool.invoke(payload)
        if inspect.isawaitable(res):
            return await res
        return res

    async def run(self, context: ExecutionContext, **global_inputs):
        await self.run_async(context, **global_inputs)
        return context.data

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
        merged["previous_id"] = list(step.inputs_from)

        
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
    
