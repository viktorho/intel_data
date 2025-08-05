import logging
import os
import urllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, AsyncGenerator, Tuple, Optional
from urllib.robotparser import RobotFileParser

import aiohttp
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai import LLMExtractionStrategy
from .search_tool.factory import build_search_tool

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
import json
from crawl4ai.chunking_strategy import (
    SlidingWindowChunking,
    FixedLengthWordChunking,
    RegexChunking,
    NlpSentenceChunking,
    TopicSegmentationChunking,
    OverlappingWindowChunking
)
from langextract.data import ExampleData
from langextract import extract
CHUNKERS = {
    "sliding": SlidingWindowChunking,
    "fixed": FixedLengthWordChunking,
    "regex": RegexChunking,
    "nlp_sentence": NlpSentenceChunking,
    "topic_segmentation": TopicSegmentationChunking,
    "overlapping_window": OverlappingWindowChunking
}


import urllib.parse as ul

class WebCrawler:
    """
    Usage:
    This class provides a web crawler that can search for URLs based on queries,
    crawl those URLs, and download images from the crawled pages.
    It uses the crawl4ai library for crawling and supports multiple search clients.
    """

    def __init__(
        self,
        conf: Dict,
        *,
        strategy: str | None = None,          
        max_workers: int = 4,                 
        langextract_sample: List["ExampleData"] | None = None,
    ):
        """
        Parameters
        ----------
        conf      : YAML text, a YAML file path, or an already-loaded dict.
        strategy  : Which strategy under `strategies` to activate.  
                    If omitted, the first one in the YAML is used.
        override  : Dict you want to force-merge on top of the selected strategy
                    (handy in unit tests or CLI flags).
        max_workers : Thread pool size for fetch & parse.
        """
        # ----------------------------------------------------------------- #
        # 1. read + validate yaml                                           #
        # ----------------------------------------------------------------- #
        print(conf)
        if conf is None:
            raise KeyError("YAML must have top-level 'web_crawler_conf'")

        shared     = conf.get("shared", {})
        strategies = {
            name: self._deep_merge(shared, spec)
            for name, spec in conf.get("strategies", {}).items()
        }
        if not strategies:
            raise ValueError("No strategies declared under 'strategies'")

        # choose active strategy
        strategy   = strategy or next(iter(strategies))
        if strategy not in strategies:
            raise ValueError(f"Strategy '{strategy}' not found in YAML")

        active = strategies[strategy]

        # ----------------------------------------------------------------- #
        # 2. materialise configs using the same helpers you already had     #
        # ----------------------------------------------------------------- #
        self.top_k        = int(active.get("top_k", 4))
        self.USER_AGENT   = active.get("user_agent", "")
        self.output_dir   = str(Path.cwd() / "output")
        self.langextract_sample = langextract_sample or []

        # keep the whole strategy map for later switching
        self._strategies = strategies
        self._current_strategy = strategy

        # initialise subsystems
        self.search_clients = self._init_search_client(active["search_client"])
        self.chunker        = self._init_chunker(active.get("chunk_config", {}))
        self.browser_config = self._init_browser_config(active.get("browser_config", {}))
        self.crawler_config = self._init_crawler_config(
            active.get("crawler_config", {}),
            active.get("llm_strategy", {})
        )

        self._robots_cache: Dict[str, "RobotFileParser | None"] = {}
        self._session = None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _init_search_client(self, search_client):
        return build_search_tool(names=search_client, top_k=self.top_k)

    def _init_chunker(self, chunk_config: Dict):
        chunking_strategy = chunk_config.get("chunking_strategy", "sliding")
        if chunking_strategy not in CHUNKERS:
            raise ValueError(f"Invalid chunking strategy: {chunking_strategy}. "
                             f"Available strategies: {list(CHUNKERS.keys())}")

        params = chunk_config.get("params", {})
        if chunking_strategy == "regex":
            import re
            params["patterns"] = [
                re.compile(p) if isinstance(p, str) else p
                for p in params.get("patterns", [])
            ]

        return CHUNKERS[chunking_strategy](**params)

    def _init_browser_config(self, browser_config: Dict):
        if browser_config is None:
            browser_config = {}
        return BrowserConfig(**browser_config)

    def _init_crawler_config(self, crawler_config: Dict, llm_strategy: Dict):
        if crawler_config is None:
            crawler_config = {}

        if "cache_mode" in crawler_config and crawler_config["cache_mode"] is not None:
            crawler_config["cache_mode"] = CacheMode[crawler_config["cache_mode"]]

        crawler_config["extraction_strategy"] = (
            LLMExtractionStrategy(**llm_strategy) if llm_strategy else None
        )

        return CrawlerRunConfig(**crawler_config)

    async def _get_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _search_urls(self, query: str) -> List[str]:
        """
        Search URLs for a list of queries using the search client.
        Args:
            query str: search query.
        Returns:
            List[str]: List of URLs found for the queries.
        """
        urls: List[str] = []
        try:
            results = await self.search_clients.search(query)
            urls_for_q = [item["url"] for item in results]
            print(urls_for_q)
            exit()
            urls.extend([u for u in urls_for_q if self._can_fetch(u)][:self.top_k])
            return list(dict.fromkeys(urls))
        except Exception as e:
            logger.error("❌ Search URLs failed for query %r: %s", query, e)

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def close_session(self):
        if self._session:
            await self._session.close()

    def _extract_features(self, chunk_text):
        result = [extract(text, examples=self.langextract_sample,
                         model="gemini-1.5-flash") for text in chunk_text]
        return result

    async def _process_single(
            self, url: str, res, download_images: bool
    ) -> Tuple[List[str], List[Dict]]:
        md = res.markdown.fit_markdown or res.markdown.raw_markdown
        # md = res
        text_chunks: List[str] = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: self._extract_features(list(self.chunker.chunk(md)))
        )
        doc_urls  = [url] * len(text_chunks)
        if download_images and res.media.get("images"):
            imgs = res.media["images"][:10]
            tasks = [
                self._download_image(img_url, page_url=url)
                for img_url in imgs
            ]
            # batching
            for i in range(0, len(tasks), 5):
                await asyncio.gather(*tasks[i:i+5], return_exceptions=True)
        
        return text_chunks


    async def _crawl_urls(self, urls: List[str],
                         download_images:bool = False
                         ) -> AsyncGenerator[List[Dict], None]:
        """
        Crawl URLs with crawl4ai.
        Args:
            urls (List[str]): List of URLs to crawl.
            download_images (bool): Whether to download images from the crawled pages.
        """
        self._session = await self._get_session()
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            if not urls:
                logger.warning("No URLs found, nothing to crawl.")
                return
            results = await crawler.arun_many(urls=urls,
                                              run_config=self.crawler_config)
            tasks = []
            for url, res in zip(urls, results):
                if isinstance(res, Exception):
                    logger.error("❌ Crawl failed %s: %s", url, res)
                else:
                    tasks.append(
                            asyncio.create_task(
                                self._process_single(url, res, download_images)
                        )
                    )
        for task in asyncio.as_completed(tasks):
            try:
                yield await task
            except Exception as e:
                logger.error("❌ Processing failed: %s", e)

    async def _download_image(self, image_list_urls: List, page_url:str = "") -> None:
        image_paths: List[str] = []
        for img in image_list_urls:
            src = img.get("src")
            if not src:
                continue
            full_url = urllib.parse.urljoin(page_url, src)
            try:
                async with self._session.get(full_url) as resp:
                    if resp.status != 200:
                        logging.warning("❗ Download failed %s: HTTP %d", full_url, resp.status)
                        continue
                    data = await resp.read()

                    path = urllib.parse.urlparse(full_url).path
                    ext = os.path.splitext(path)[1] or ".jpg"
                    filename = os.path.join(self.output_dir,f"./images/{abs(hash(full_url))}{ext}")
                    os.makedirs(os.path.dirname(filename), exist_ok=True)

                    with open(filename, "wb") as f:
                        f.write(data)
                    image_paths.append(filename)
            except Exception as e:
                logging.error("❗ Download Exception  %s: %s", full_url, e)

    async def crawl_queries(self, queries: List[str], download_image: bool=False) ->\
            AsyncGenerator[List[Dict], None]:
        """
        Flow:
          queries → search_urls → crawl_urls → return docs
        """
        if not queries:
            logger.warning("No queries provided, nothing to crawl.")
            return
        for q in queries:
            urls = await self._search_urls(q)
            
            async for chunk in self._crawl_urls(urls,download_images=download_image):
                yield chunk

    async def save_output(self, queries: List[str], save_path: str = "output.txt", need_other_datatype: Optional[List[str]] = None):
        """
        Runs crawl_queries and saves the results to a text file (one line per document as JSON).
        """
        output_file = Path(save_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent dir exists
        i = 0
        with output_file.open("w", encoding="utf-8") as f:
            async for chunk in self.crawl_queries(queries, need_other_datatype):
                i+=1
                for doc in chunk:
                    json_line = json.dumps(doc, ensure_ascii=False)
                    f.write(json_line + "\n")
                f.write(f"=========================={i}==========================+\n\n")
    def _can_fetch(
            self,
            url: str,
    ) -> bool:
        """
        Check if the crawler can fetch a URL based on the robots.txt rules.
        Args:
            url (str): The URL to check.
        Returns:
            bool: True if the crawler can fetch the URL, False otherwise.
        """
        origin = f"{ul.urlparse(url).scheme}://{ul.urlparse(url).netloc}"
        rp = self._robots_cache.get(origin)
        if rp is None:
            rp = RobotFileParser()
            rp.set_url(ul.urljoin(origin, "/robots.txt"))
            try:
                rp.read()
            except Exception:
                rp = None
            self._robots_cache[origin] = rp
        return True if rp is None else rp.can_fetch(self.USER_AGENT, url)

async def main():
    text = ("""
        ## Downloadable Housing Market Data From Redfin
        Redfin is a real estate brokerage, meaning we have direct access to data from local multiple listing services, as well as insight from our real estate agents across the country. That’s why we’re able to give you the [earliest and most reliable data](https://www.redfin.com/about/data-quality-on-redfin) on the state of the housing market. We publish existing industry data faster, and offer additional data on tours and offers that no one else has. Using the tools below, you can visualize and download housing market data for metropolitan areas, cities, neighborhoods and zip codes across the nation. You may learn how to use the tools in this [video tutorial](https://drive.google.com/file/d/1GRoE9MvjXIf-kfLhvfM4-URcu8kr9QDN/view?usp=sharing).
        ## Redfin Weekly Housing Market Data
        This weekly data will be updated every Wednesday with new data for the prior week.
        All data here is computed daily as either a rolling 1, 4 or 12-week window. The local data is grouped by metropolitan area and by county. All of this data is subject to revisions weekly and should be viewed with caution. If there are any concerns about the data or questions about metric definitions, please e-mail econdata@redfin.com or press@redfin.com. You can download the full dataset [here](https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_covid19/weekly_housing_market_data_most_recent.tsv000.gz). For more information, we’ve compiled all the [definitions for each metric](https://www.redfin.com/news/data-center-metrics-definitions/).
        ## Redfin Monthly Housing Market Data
        To view additional housing market data at the local level, please visit our U.S. Housing Market Overview page here: <https://www.redfin.com/us-housing-market>
        """)
    await crawler._process_single(url="sada.com", res=text, download_images=False)


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parents[2] / ".env"

    load_dotenv(env_path)
    logging.basicConfig(level=logging.INFO)
    crawler = WebCrawler(
        search_client="brave",
        top_k=5,
        browser_config={
            "headless": True,
        },
        crawler_config={
            "wait_until": "networkidle",
            "cache_mode": "BYPASS",
            "page_timeout": 90000,
            "css_selector": "article, main",
            "excluded_tags": ["nav", "footer", "form", "aside"],
            "exclude_external_links": True,
            "remove_overlay_elements": True
        },
        chunk_config = {
            "chunking_strategy": "regex",
            "params": {
                "patterns": [r"(?<![A-Z]\.[A-Z]\.)(?<![A-Z][a-z]\.)(?<=[.?!])\s+"]
            }
}
    )

    asyncio.run(main())