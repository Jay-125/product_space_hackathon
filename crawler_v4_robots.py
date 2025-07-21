import sys
import re
import os
import asyncio
import requests
from typing import List, Dict, Any
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from urllib.robotparser import RobotFileParser
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from utils import get_chroma_client, get_or_create_collection, add_documents_to_collection


# ------------- Utility Functions -------------

def get_disallowed_paths_from_robots_txt(base_url: str) -> List[str]:
    """Fetch and parse robots.txt to get disallowed paths."""
    parsed_url = urlparse(base_url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    try:
        resp = requests.get(robots_url)
        if resp.status_code != 200:
            print(f"robots.txt not found at {robots_url}")
            return []
        lines = resp.text.splitlines()
    except Exception as e:
        print(f"Error fetching robots.txt: {e}")
        return []

    disallowed = []
    user_agent = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("user-agent"):
            user_agent = line.split(":", 1)[1].strip()
        elif line.lower().startswith("disallow") and (user_agent == "*" or user_agent is None):
            path = line.split(":", 1)[1].strip()
            if path:
                disallowed.append(path)
    return disallowed

def is_sitemap(url: str) -> bool:
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    return url.endswith('.txt')

def smart_chunk_markdown(markdown: str, max_len: int = 1000) -> List[str]:
    def split_by_header(md, header_pattern):
        indices = [m.start() for m in re.finditer(header_pattern, md, re.MULTILINE)]
        indices.append(len(md))
        return [md[indices[i]:indices[i+1]].strip() for i in range(len(indices)-1) if md[indices[i]:indices[i+1]].strip()]

    chunks = []

    for h1 in split_by_header(markdown, r'^# .+$'):
        if len(h1) > max_len:
            for h2 in split_by_header(h1, r'^## .+$'):
                if len(h2) > max_len:
                    for h3 in split_by_header(h2, r'^### .+$'):
                        if len(h3) > max_len:
                            for i in range(0, len(h3), max_len):
                                chunks.append(h3[i:i+max_len].strip())
                        else:
                            chunks.append(h3)
                else:
                    chunks.append(h2)
        else:
            chunks.append(h1)

    final_chunks = []
    for c in chunks:
        if len(c) > max_len:
            final_chunks.extend([c[i:i+max_len].strip() for i in range(0, len(c), max_len)])
        else:
            final_chunks.append(c)

    return [c for c in final_chunks if c]

def extract_section_info(chunk: str) -> Dict[str, Any]:
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''
    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

def parse_sitemap(sitemap_url: str) -> List[str]:
    resp = requests.get(sitemap_url)
    urls = []
    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")
    return urls

# ------------- Crawl Functions -------------

async def crawl_recursive_internal_links(start_urls, max_depth=3, max_concurrent=10, disallowed_paths=[]) -> List[Dict[str, Any]]:
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()
    def normalize_url(url):
        return urldefrag(url)[0]

    def is_allowed(url):
        parsed = urlparse(url)
        return not any(parsed.path.startswith(dp) for dp in disallowed_paths)

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for depth in range(max_depth):
            urls_to_crawl = [url for url in current_urls if url not in visited and is_allowed(url)]
            if not urls_to_crawl:
                break

            results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
            next_level_urls = set()

            for result in results:
                norm_url = normalize_url(result.url)
                visited.add(norm_url)

                if result.success and result.markdown:
                    results_all.append({'url': result.url, 'markdown': result.markdown})
                    for link in result.links.get("internal", []):
                        next_url = normalize_url(link["href"])
                        if next_url not in visited and is_allowed(next_url):
                            next_level_urls.add(next_url)

            current_urls = next_level_urls

    return results_all

async def crawl_markdown_file(url: str) -> List[Dict[str,Any]]:
    browser_config = BrowserConfig(headless=True)
    crawl_config = CrawlerRunConfig()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)
        if result.success and result.markdown:
            return [{'url': url, 'markdown': result.markdown}]
        else:
            print(f"Failed to crawl {url}: {result.error_message}")
            return []

async def crawl_batch(urls: List[str], max_concurrent: int = 10, disallowed_paths: List[str] = []) -> List[Dict[str,Any]]:
    browser_config = BrowserConfig(headless=True, verbose=False)
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    def is_allowed(url):
        parsed = urlparse(url)
        return not any(parsed.path.startswith(dp) for dp in disallowed_paths)

    urls = [url for url in urls if is_allowed(url)]

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
        return [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]

# ------------- Main -------------

def main(url):
    # 🔧 Define input variables here
    # url = "https://theproductspace.in/sitemap.xml"  # Replace with your URL
    url = url
    collection = "my_company_collection"
    db_dir = "./my_company_chroma"
    embedding_model = "all-MiniLM-L6-v2"
    chunk_size = 1000
    max_depth = 2
    max_concurrent = 5
    batch_size = 50

    disallowed_paths = get_disallowed_paths_from_robots_txt(url)

    # Crawl based on input type
    if is_txt(url):
        print(f"Detected .txt/markdown file: {url}")
        crawl_results = asyncio.run(crawl_markdown_file(url))

    elif is_sitemap(url):
        print(f"Detected sitemap: {url}")
        sitemap_urls = parse_sitemap(url)
        if not sitemap_urls:
            print("No URLs found in sitemap.")
            sys.exit(1)
        crawl_results = asyncio.run(
            crawl_batch(sitemap_urls, max_concurrent=max_concurrent, disallowed_paths=disallowed_paths)
        )

    else:
        print(f"Detected regular URL: {url}")
        crawl_results = asyncio.run(
            crawl_recursive_internal_links(
                [url], max_depth=max_depth, max_concurrent=max_concurrent, disallowed_paths=disallowed_paths
            )
        )

    # Chunk and insert into Chroma
    ids, documents, metadatas = [], [], []
    chunk_idx = 0
    for doc in crawl_results:
        source_url = doc['url']
        md = doc['markdown']
        chunks = smart_chunk_markdown(md, max_len=chunk_size)
        for chunk in chunks:
            ids.append(f"chunk-{chunk_idx}")
            documents.append(chunk)


if __name__ == "__main__":
    main("https://theproductspace.in/sitemap.xml")