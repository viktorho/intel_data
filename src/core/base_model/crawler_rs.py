from __future__ import annotations

from .prompt_rs import ListPromptFm
from pydantic import BaseModel
from typing import List, Any, ClassVar, Type
from core.subtools import WebCrawler
from langextract.data import ExampleData
from core.helper import inject_helper
from core.subtools import WebCrawler, BasicLangExtractor


@inject_helper({"crawler": "WebCrawler",
                })
async def crawl_data(
        *,
        _input: CrawlerFm,
        crawler: WebCrawler, 
    ):
    list_queries = _input.subprompt.to_str_queries()
    async for q in crawler.crawl_queries(list_queries):
        return True


@inject_helper({
                "lang_extractor": "BasicLangExtractor"
                })
async def parse_json(
        *,
        _input: LangExtractFm,
        lang_extractor: BasicLangExtractor
    ):
    example_data = _input.example_data
    


class CrawlerFm(BaseModel):
    HELPER: ClassVar[List[Type[Any]]] = [WebCrawler]
    batch_size: str = 5
    subprompt: ListPromptFm

class LangExtractFm(BaseModel):
    HELPER: ClassVar[List[Type[Any]]] = [BasicLangExtractor]

    example_data: ExampleData
    