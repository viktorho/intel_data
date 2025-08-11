from re import sub
from typing import List,Tuple,Union
import networkx as nx

from collections import defaultdict


from .core.base_model import *
from .core.agent_tools import tool_factory
from langchain_core.language_models.chat_models import BaseChatModel
from .core.dag import StepSpec,PlanSpec, DAGBuilder,\
                    LLMDAGExecutor, registry_tools, ExecutionContext
from db import DataTableManager
from typing import Dict
from project_setup import load_config

class DefaultPlannerPipeline:
    """
    End-to-end pipeline: extract RequirementFm ► clarify (if needed) ► generate features.
    """

    def __init__(self, 
                 llm: Union[BaseChatModel,List[BaseChatModel]],
                 helper_cfg: Dict[str, str],
                 ):
        tool_llm = llm[0] if isinstance(llm, list) else llm
        self.classes = self._initialize_helper(helper_cfg)
        self._define_tools(tool_llm=tool_llm)
        
    @staticmethod
    def _initialize_helper(conf: Dict[str,str]) -> Dict:
        helper_classes = defaultdict(dict)

        for c in conf:
            keys = list(c.keys())
            path = keys[1] if keys[0] == "type" else keys[0]
            _temp = c["type"].split("/")
            class_name = _temp[1]
            tool_name = _temp[0]
            other_value = c[path]
            helper_classes[tool_name][class_name] = load_config(other_value)
            
        return helper_classes

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
        self.crawler_tool = tool_factory(
            schema=CrawlerFm,
            name="crawler_tool",
            description="Crawl data from subprompt generator",
            default_helper_kwargs=self.classes.get("CrawlerToolFm"),
            post_hook=crawl_data,
        )
        self.langextract_tool = tool_factory(
            schema=LangExtractFm,
            name="langextract_tool",
            description="Crawl data from subprompt generator",
            post_hook=parse_json,
        )
        self.genexample_tool = tool_factory(
            llm=tool_llm,
            schema=ExampleDataFm,
            name="genexample_tool",
            description="Generate example for extracting features",
            post_hook=to_langextract
        )
        self.registry = registry_tools(self)

    ###================================= TESTING PLAN =================================###

    def research_plan(self) -> PlanSpec:
        return PlanSpec(
            plan_id= "research-00",
            description= "Default research plan",
            steps=[
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
        return PlanSpec(
            plan_id= "process_data-00",
            description= "Default process data plan",
            steps=[
            StepSpec(
                id="gen_example",
                tool_name="genexample_tool",
                description="Define features to start extracting",
                inputs_from=[],
                input_key_map={"req": "__GLOBAL__", 
                               "topic": "__GLOBAL__",
                               },
            ),
            StepSpec(
                id="crawl_feature",
                tool_name="crawler_tool",
                description="Define fetures to start ",
                inputs_from=[],
                input_key_map={
                    "subprompt": "__GLOBAL__",
                    "save_dir": "__GLOBAL__",
                }
            ),
            StepSpec(
                id="lang_extract",
                tool_name="langextract_tool",
                description="Extract json features from docs",
                inputs_from=["crawl_feature", "gen_example"]
            )
        ])
    
    def fill_table(self, table: DataTableManager) -> nx.DiGraph:
        pass
    

    def _prepare_plan(self, plan: PlanSpec):
        dag = DAGBuilder(plan=plan).build()
        executor = LLMDAGExecutor(dag, self.registry)
        return executor
    
    async def run(self, plan: PlanSpec, *, branch_id: str = "default", run_id: str | None = None, **kwargs):
        import uuid
        executor = self._prepare_plan(plan)
        run_id = run_id or uuid.uuid4().hex[:8]
        context = ExecutionContext(
            plan_id=plan.plan_id,
            branch_id=branch_id,
            run_id=run_id,
            plan_spec=plan,
        )
        await executor.run(context, **kwargs)
        return context.data

