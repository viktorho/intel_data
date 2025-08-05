from re import sub
from tokenize import Triple
from typing import List,Tuple,Union
import networkx as nx


from .base_model import *
from .tools import tool_factory
from langchain_core.language_models.chat_models import BaseChatModel
from .dag import StepSpec,PlanSpec, DAGBuilder,\
                    LLMDAGExecutor, registry_tools, ExecutionContext
from db import DataTableManager



class P1:
    """
    End-to-end pipeline: extract RequirementFm ► clarify (if needed) ► generate features.
    """



    def __init__(self, llm: Union[BaseChatModel,List[BaseChatModel]], nofsub: int = 5):
        tool_llm = llm[0] if isinstance(llm, list) else llm
        self.nofsub = nofsub
        self._define_tools(tool_llm=tool_llm)
        
        
    def _define_tools(self, tool_llm):
        self.extract_reqs_tool = tool_factory(
            llm=tool_llm,
            schema=RequirementFm,
            name="extract_reqs_tool",
            # post_hook=clarify_requirement,
            description="Extract core information to make sure having enough information from user request",
        )
        self.feature_tool = tool_factory(
            llm=tool_llm,
            schema=FeaturesFm,
            name="feature_tool",
            description="Generate candidate features from RequirementFm",
        )
        self.subprompt_tool = tool_factory(
            llm=tool_llm,
            schema=ListPromptFm,
            name="subprompt_tool",
            description="Getting understand about the topic"
        )
        self.genexample_tool = tool_factory(
            llm=tool_llm,
            schema=ExampleDataFm,
            pre_hook=[feature_filter],
            name="genexample_tool",
            description="Generate example for extracting features",
            post_hook=[to_langextract]
        )
        self.registry = registry_tools(self)

    ###================================= TESTING PLAN ===============================================

    def research_plan(self) -> PlanSpec:
        return PlanSpec(steps=[
            StepSpec(
                id="extract",
                tool_name="extract_reqs_tool",
                description="Read the user request and extract the main topic or intent.",
                inputs_from=[],
                input_key_map={"req": "__GLOBAL__"},
                
            ),
            StepSpec(
                id="subprompt",
                tool_name="subprompt_tool",
                description="Break down the main request into smaller, more specific sub-questions.",
                inputs_from=["extract"],
                input_key_map={"req": "extract", "nofsub":"__GLOBAL__"},
            ),
            StepSpec(
                id="feature",
                tool_name="feature_tool",
                description="Analyze the extracted request and generate useful features or attributes for it (e.g. key terms or tags).",
                inputs_from=["extract"],
                input_key_map={"req":"extract"}
            )
            ]
        ) 
    
    def process_data(self) -> PlanSpec:
        return PlanSpec(steps=[
            StepSpec(
                id="gen_example",
                tool_name="genexample_tool",
                description="Define features to start extracting",
                inputs_from=[],
                input_key_map={"req": "__GLOBAL__", 
                               "topic": "__GLOBAL__"},
            ),
        ])
    
    def fill_table(self, table: DataTableManager) -> nx.DiGraph:
        pass
    

    def _prepare_plan(self, plan: PlanSpec):
        dag = DAGBuilder(plan=plan).build()
        executor = LLMDAGExecutor(dag, self.registry)
        return executor
    
    def run(self,plan: PlanSpec, **kwargs):
        executor = self._prepare_plan(plan)
        context = ExecutionContext()
        executor.run(
            context,
            **kwargs,
        )
        return context.data
    

