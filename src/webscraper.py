import os
import re
import json
import uuid
import time
import logging
import asyncio
import requests
import tiktoken
import fitz
from tqdm import tqdm
from datetime import datetime
from urllib.parse import urljoin

from utils.url_utils import normalize_url, is_html_page
from utils.file_utils import write_jsonl_line

class WebPDFScraper:
    def __init__(self, output_file="mosdac_pdfs.jsonl", text_output_file="mosdac_pdfs_text.jsonl",
                 llm_output_file="llm_ready_output.jsonl", download_folder="downloaded_pdfs", log_dir="logs",
                 max_depth=3, max_concurrent=10):
        self.output_file = output_file
        self.text_output_file = text_output_file
        self.llm_output_file = llm_output_file
        self.download_folder = download_folder
        self.max_depth = max_depth
        self.max_concurrent = max_concurrent
        self.visited = set()
        self.pdf_links = set()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"crawl_{timestamp}.log"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )

        try:
            from crawl4ai import (
                AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode,
                MemoryAdaptiveDispatcher
            )
            self.AsyncWebCrawler = AsyncWebCrawler
            self.BrowserConfig = BrowserConfig
            self.CrawlerRunConfig = CrawlerRunConfig
            self.CacheMode = CacheMode
            self.MemoryAdaptiveDispatcher = MemoryAdaptiveDispatcher
        except ImportError:
            logging.error("crawl4ai is not installed. Please install it to run the scraper.")
            raise

    async def extract_pdfs(self, start_urls):
        current_urls = {normalize_url(u) for u in start_urls}

        browser_config = self.BrowserConfig(headless=True, verbose=False)
        run_config = self.CrawlerRunConfig(cache_mode=self.CacheMode.BYPASS, stream=False)
        dispatcher = self.MemoryAdaptiveDispatcher(memory_threshold_percent=70.0, check_interval=1.0,
                                                    max_session_permit=self.max_concurrent)

        for depth in range(self.max_depth):
            logging.info(f"=== Crawling Depth {depth + 1} ===")
            urls_to_crawl = [url for url in current_urls if url not in self.visited and is_html_page(url)]
            if not urls_to_crawl:
                break

            async with self.AsyncWebCrawler(config=browser_config) as crawler:
                results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)

            next_level_urls = set()
            for result in results:
                try:
                    norm_url = normalize_url(result.url)
                    self.visited.add(norm_url)

                    if not result.success or not result.links:
                        continue

                    all_links = result.links.get("internal", []) + result.links.get("external", [])
                    for link in all_links:
                        href = link.get("href")
                        if not href:
                            continue
                        full_url = normalize_url(urljoin(result.url, href))
                        if href.endswith(".pdf"):
                            if full_url not in self.pdf_links:
                                self.pdf_links.add(full_url)
                                logging.info(f"[PDF] {full_url}")
                                write_jsonl_line(self.output_file, {
                                    "pdf_url": full_url,
                                    "source_page": result.url
                                })
                        else:
                            if full_url not in self.visited:
                                next_level_urls.add(full_url)
                except Exception as e:
                    logging.warning(f"Error processing result from {result.url}: {e}")

            current_urls = next_level_urls

    def download_and_extract(self):
        os.makedirs(self.download_folder, exist_ok=True)
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                urls = [json.loads(line)["pdf_url"] for line in f]
        except FileNotFoundError:
            logging.warning(f"{self.output_file} not found. Skipping download.")
            return

        with open(self.text_output_file, "w", encoding="utf-8") as out_file:
            for url in tqdm(urls, desc="Processing PDFs", unit="file"):
                try:
                    filename = os.path.join(self.download_folder, url.split("/")[-1])
                    if not os.path.exists(filename):
                        r = requests.get(url, timeout=30)
                        with open(filename, "wb") as f:
                            f.write(r.content)

                    doc = fitz.open(filename)
                    text = "".join(page.get_text() for page in doc)
                    tokens = self.count_tokens(text)

                    out_file.write(json.dumps({
                        "id": str(uuid.uuid4()),
                        "url": url,
                        "title": os.path.basename(filename),
                        "text": text,
                        "tokens": tokens
                    }, ensure_ascii=False) + "\n")
                except Exception as e:
                    tqdm.write(f"[ERROR] Failed processing {url}: {e}")

    def preprocess_for_llm(self, input_file=None, output_file=None):
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        input_file = input_file or self.text_output_file
        output_file = output_file or self.llm_output_file

        def clean_markdown(md: str) -> str:
            md = re.sub(r'!\[.*?\]\(.*?\)', '', md)
            md = re.sub(r'\[.*?\]\(javascript:[^)]+\)', '', md)
            md = re.sub(r'\[\s*\]\(.*?\)', '', md)
            md = re.sub(r'\[.*?\]\(.*?\)', lambda m: m.group(0).split(']')[0][1:], md)
            md = re.sub(r'[#`\\*]{1,}', '', md)
            md = re.sub(r'\n{2,}', '\n\n', md)
            return md.strip()

        def extract_sections(md: str):
            sections = []
            current = {"heading": None, "content": []}
            for line in md.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if current["heading"] and current["content"]:
                        sections.append(current)
                    current = {"heading": line.lstrip("#").strip(), "content": []}
                else:
                    current["content"].append(line)
            if current["heading"] and current["content"]:
                sections.append(current)
            return sections or [{"heading": "General", "content": md.splitlines()}]

        def chunk_text(text: str, max_tokens=512):
            words = text.split()
            chunks = []
            current = []
            for word in words:
                current.append(word)
                token_count = len(enc.encode(" ".join(current)))
                if token_count >= max_tokens:
                    chunks.append(" ".join(current))
                    current = []
            if current:
                chunks.append(" ".join(current))
            return chunks

        with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
            for line in infile:
                try:
                    data = json.loads(line)
                    url = data.get("url")
                    content = data.get("text", "")
                    if not content or not url:
                        continue
                    cleaned = clean_markdown(content)
                    sections = extract_sections(cleaned)
                    for section in sections:
                        text = " ".join(section["content"])
                        chunks = chunk_text(text)
                        for i, chunk in enumerate(chunks):
                            record = {
                                "id": f"{url}#chunk-{i+1}",
                                "url": url,
                                "title": section["heading"],
                                "text": chunk,
                                "tokens": len(enc.encode(chunk))
                            }
                            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as e:
                    logging.warning(f"Error processing line: {e}")

    def count_tokens(self, text: str) -> int:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(text))



# import asyncio
# import os
# import logging
# from urllib.parse import urljoin
# from datetime import datetime

# from utils.url_utils import normalize_url, is_html_page
# from utils.file_utils import write_jsonl_line


# class WebPDFScraper:
#     def __init__(self, output_file="mosdac_pdfs.jsonl", log_dir="logs", max_depth=3, max_concurrent=10):
#         self.output_file = output_file
#         self.max_depth = max_depth
#         self.max_concurrent = max_concurrent
#         self.visited = set()
#         self.pdf_links = set()

#         # Logging setup
#         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         log_file = f"crawl_{timestamp}.log"
#         os.makedirs(log_dir, exist_ok=True)
#         log_path = os.path.join(log_dir, log_file)

#         logging.basicConfig(
#             level=logging.INFO,
#             format="%(asctime)s - %(levelname)s - %(message)s",
#             datefmt="%Y-%m-%d %H:%M:%S",
#             handlers=[
#                 logging.FileHandler(log_path),
#                 logging.StreamHandler()
#             ]
#         )

#         try:
#             from crawl4ai import (
#                 AsyncWebCrawler,
#                 BrowserConfig,
#                 CrawlerRunConfig,
#                 CacheMode,
#                 MemoryAdaptiveDispatcher
#             )
#             self.AsyncWebCrawler = AsyncWebCrawler
#             self.BrowserConfig = BrowserConfig
#             self.CrawlerRunConfig = CrawlerRunConfig
#             self.CacheMode = CacheMode
#             self.MemoryAdaptiveDispatcher = MemoryAdaptiveDispatcher
#         except ImportError:
#             logging.error("crawl4ai is not installed. Please install it to run the scraper.")
#             raise

#     async def run(self, start_urls):
#         current_urls = {normalize_url(u) for u in start_urls}

#         browser_config = self.BrowserConfig(headless=True, verbose=False)
#         run_config = self.CrawlerRunConfig(cache_mode=self.CacheMode.BYPASS, stream=False)
#         dispatcher = self.MemoryAdaptiveDispatcher(
#             memory_threshold_percent=70.0,
#             check_interval=1.0,
#             max_session_permit=self.max_concurrent
#         )

#         for depth in range(self.max_depth):
#             logging.info(f"=== Crawling Depth {depth + 1} ===")
#             all_urls = {u for u in current_urls if u not in self.visited}
#             urls_to_crawl = [url for url in all_urls if is_html_page(url)]

#             if not urls_to_crawl:
#                 break

#             async with self.AsyncWebCrawler(config=browser_config) as crawler:
#                 results = await crawler.arun_many(
#                     urls=urls_to_crawl,
#                     config=run_config,
#                     dispatcher=dispatcher
#                 )

#             next_level_urls = set()

#             for result in results:
#                 try:
#                     norm_url = normalize_url(result.url)
#                     self.visited.add(norm_url)

#                     if not result.success or not result.links:
#                         continue

#                     all_links = result.links.get("internal", []) + result.links.get("external", [])
#                     for link in all_links:
#                         href = link.get("href")
#                         if not href:
#                             continue

#                         full_url = normalize_url(urljoin(result.url, href))

#                         if href.endswith(".pdf"):
#                             if full_url not in self.pdf_links:
#                                 self.pdf_links.add(full_url)
#                                 logging.info(f"[PDF] {full_url}")
#                                 write_jsonl_line(self.output_file, {
#                                     "pdf_url": full_url,
#                                     "source_page": result.url
#                                 })
#                         else:
#                             if full_url not in self.visited:
#                                 next_level_urls.add(full_url)

#                 except Exception as e:
#                     logging.warning(f"Error processing result from {result.url}: {e}")

#             current_urls = next_level_urls



# import asyncio
# from src.webscraper import WebPDFScraper

# if __name__ == "__main__":
#     seed_urls = ["https://example.com"]
#     scraper = WebPDFScraper()
#     asyncio.run(scraper.run(seed_urls))



# # import asyncio
# # import os
# # import logging
# # from urllib.parse import urljoin

# # from utils.url_utils import normalize_url, is_html_page
# # from utils.file_utils import write_jsonl_line
# # from datetime import datetime

# # LOG_DIR = "logs"
# # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# # LOG_FILE = f"crawl_{timestamp}.log"
# # LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# # os.makedirs(LOG_DIR, exist_ok=True)

# # # Setup logging
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format="%(asctime)s - %(levelname)s - %(message)s",
# #     handlers=[
# #         logging.FileHandler(os.path.join(LOG_DIR, LOG_FILE)),
# #         logging.StreamHandler()  # prints to stdout
# #     ]
# # )



# # async def extract_pdfs(start_urls, max_depth=3, max_concurrent=10):
    
# #     PDF_OUTPUT = "mosdac_pdfs.jsonl"
# #     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
# #     try:
# #         from crawl4ai import (
# #             AsyncWebCrawler,
# #             BrowserConfig,
# #             CrawlerRunConfig,
# #             CacheMode,
# #             MemoryAdaptiveDispatcher
# #         )
# #     except ImportError:
# #         logging.error("crawl4ai is not installed. Please install it to run the scraper.")
# #         return

# #     browser_config = BrowserConfig(headless=True, verbose=False)
# #     run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
# #     dispatcher = MemoryAdaptiveDispatcher(
# #         memory_threshold_percent=70.0,
# #         check_interval=1.0,
# #         max_session_permit=max_concurrent
# #     )

# #     visited = set()
# #     pdf_links = set()
# #     current_urls = {normalize_url(u) for u in start_urls}

# #     for depth in range(max_depth):
# #         logging.info(f"=== Crawling Depth {depth+1} ===")

# #         all_urls = {u for u in current_urls if u not in visited}
# #         urls_to_crawl = [url for url in all_urls if is_html_page(url)]

# #         if not urls_to_crawl:
# #             break

# #         async with AsyncWebCrawler(config=browser_config) as crawler:
# #             results = await crawler.arun_many(
# #                 urls=urls_to_crawl,
# #                 config=run_config,
# #                 dispatcher=dispatcher
# #             )

# #         next_level_urls = set()

# #         for result in results:
# #             try:
# #                 norm_url = normalize_url(result.url)
# #                 visited.add(norm_url)

# #                 if not result.success or not result.links:
# #                     continue

# #                 all_links = result.links.get("internal", []) + result.links.get("external", [])
# #                 for link in all_links:
# #                     href = link.get("href")
# #                     if not href:
# #                         continue

# #                     full_url = normalize_url(urljoin(result.url, href))

# #                     if href.endswith(".pdf"):
# #                         if full_url not in pdf_links:
# #                             pdf_links.add(full_url)
# #                             logging.info(f"[PDF] {full_url}")
# #                             write_jsonl_line(PDF_OUTPUT, {
# #                                 "pdf_url": full_url,
# #                                 "source_page": result.url
# #                             })
# #                     else:
# #                         if full_url not in visited:
# #                             next_level_urls.add(full_url)

# #             except Exception as e:
# #                 logging.warning(f"Error processing result from {result.url}: {e}")

# #         current_urls = next_level_urls


# # if __name__ == "__main__":
# #     # Replace with your actual seed URLs
# #     seed_urls = ["https://example.com"]
# #     asyncio.run(extract_pdfs(seed_urls))



# # # import asyncio
# # # import os
# # # from utils.url_utils import normalize_url, is_html_page
# # # from utils.file_utils import write_jsonl_line
# # # from utils.log_utils import log_info

# # # PDF_OUTPUT = "mosdac_pdfs.jsonl"

# # # async def extract_pdfs(start_urls, max_depth=3, max_concurrent=10):
# # #     try:
# # #         from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
# # #     except ImportError:
# # #         log_info("crawl4ai is not installed. Please install it to run the scraper.")
# # #         return

# # #     browser_config = BrowserConfig(headless=True, verbose=False)
# # #     run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
# # #     dispatcher = MemoryAdaptiveDispatcher(memory_threshold_percent=70.0, check_interval=1.0, max_session_permit=max_concurrent)

# # #     visited = set()
# # #     pdf_links = set()
# # #     current_urls = set([normalize_url(u) for u in start_urls])

# # #     for depth in range(max_depth):
# # #         log_info(f"=== Crawling Depth {depth+1} ===")
# # #         all_urls = set([normalize_url(url) for url in current_urls if normalize_url(url) not in visited])
# # #         urls_to_crawl = [url for url in all_urls if is_html_page(url)]
# # #         async with AsyncWebCrawler(config=browser_config) as crawler:
# # #             results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
# # #         next_level_urls = set()
# # #         for result in results:
# # #             norm_url = normalize_url(result.url)
# # #             visited.add(norm_url)
# # #             if result.success:
# # #                 links = result.links.get("internal", []) + result.links.get("external", [])
# # #                 for link in links:
# # #                     href = link["href"]
# # #                     if href.endswith(".pdf"):
# # #                         pdf_url = normalize_url(href)
# # #                         if pdf_url not in pdf_links:
# # #                             pdf_links.add(pdf_url)
# # #                             log_info(f"[PDF] {pdf_url}")
# # #                             write_jsonl_line(PDF_OUTPUT, {"pdf_url": pdf_url, "source_page": result.url})
# # #                 for link in result.links.get("internal", []):
# # #                     next_url = normalize_url(link["href"])
# # #                     if next_url not in visited:
# # #                         next_level_urls.add(next_url)
# # #         current_urls = next_level_urls

# # # # TODO: Integrate with a knowledge graph after PDF extraction 