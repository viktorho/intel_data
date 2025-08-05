import argparse
import sys
from uuid import UUID
import asyncio


from dotenv import load_dotenv
from core.pipeline import P1
from project_setup import build_clients, load_config, setup_logging, to_json
from core.base_model import FeaturesFm,ListPromptFm, RequirementFm
from db import DataTableManager
import json
from project_setup import build_crawler,log_value

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
    crawler = build_crawler()
    ## phase 1
    pipeline = P1(llm=llm)
    save_path = "./output/output.txt"

    data = pipeline.run(
        pipeline.research_plan(),
        req=prompt,
        nofsub=5
    )
    
    await phase2(save_path,**data)

    example = pipeline.run(pipeline.process_data(),
                           req=data.get("feature"),
                           topic=data.get("extract").topic)


async def phase2(
        save_path="",
        *,
        feature: FeaturesFm,
        subprompt: ListPromptFm,
        extract: RequirementFm
        ):
    
    db = DataTableManager(feature)

    crawler = build_crawler()
    sup_queries = [q.query for q in subprompt.queries]
    await crawler.save_output(queries=sup_queries,save_path=save_path)

if __name__ == "__main__":
    save_path = asyncio.run(main())
