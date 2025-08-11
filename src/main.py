import argparse
import sys
from uuid import UUID
import asyncio


from dotenv import load_dotenv
from pipeline import DefaultPlannerPipeline 
from project_setup import build_clients, load_config, setup_logging, to_json
from core.base_model import FeaturesFm,ListPromptFm, RequirementFm
from db import DataTableManager
import json

async def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: prompt → features → plan"
    )
    parser.add_argument(
        "--prompt",
        help="Free-form description of what to crawl"
    )
    # parser.add_argument(

    # )
    args = parser.parse_args()
    prompt = args.prompt

    setup_logging()
    load_dotenv(".env")
    conf = load_config("conf/client.yaml")
    llm = build_clients(conf.get("clients_conf"))
    
    ## phase 1
    pipeline = DefaultPlannerPipeline (
        llm=llm,
        helper_cfg=conf.get("helper_conf")
    )
    data = await pipeline.run(
        pipeline.research_plan(),
        req=prompt,
        nofsub=5
    )

    example = await pipeline.run(pipeline.process_data(),
                           req=data.get("feature"),
                           topic=data.get("extract").topic,
                           subprompt=data.get("subprompt"),
                           save_dir="output"
                           )
    print(example)



if __name__ == "__main__":
    asyncio.run(main())
