"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://admissions.vinuni.edu.vn/vi/hoc-phi/cu-nhan/",
    "https://admissions.vinuni.edu.vn/vi/hoc-phi/sau-dai-hoc/",
    "https://admissions.vinuni.edu.vn/vi/hoc-bong-ho-tro-tai-chinh/",
    "https://vinuni.edu.vn/vi/hoc-thac-si-bao-nhieu-nam/",
    "https://admissions.vinuni.edu.vn/vi/dai-hoc/ung-tuyen-vao-vinuni/ung-vien-nam-nhat/quy-trinh-ung-tuyen/"    
]


def _extract_via_requests(url: str) -> dict:
    """Fallback extraction using requests and html parsing."""
    import re
    from datetime import datetime
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=25)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)
    if not title:
        title = url.rstrip("/").split("/")[-1].replace("-", " ").title()

    # Remove script, style, nav, footer
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Extract content
    main_el = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|post|entry|main")) or soup.body
    lines = []
    if main_el:
        for p in main_el.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            text = p.get_text(strip=True)
            if not text:
                continue
            if p.name in ["h1", "h2", "h3", "h4"]:
                level = int(p.name[1])
                lines.append(f"\n{'#' * level} {text}\n")
            elif p.name == "li":
                lines.append(f"- {text}")
            else:
                lines.append(f"\n{text}\n")
    markdown_text = "\n".join(lines).strip()
    if len(markdown_text) < 200:
        raw_text = soup.get_text(separator="\n", strip=True)
        markdown_text = raw_text

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": markdown_text,
    }


async def crawl_article(url: str) -> dict:
    """Crawl a URL using Crawl4AI with fallback to requests."""
    from datetime import datetime

    # 1. Try Crawl4AI
    try:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            title = (result.metadata or {}).get("title", "").strip() if result.metadata else ""
            markdown = (result.markdown or "").strip()
            if len(markdown) >= 200:
                if not title or title.lower() in {"unknown", "none"}:
                    title = url.rstrip("/").split("/")[-1].replace("-", " ").title()
                return {
                    "url": url,
                    "title": title,
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": markdown,
                }
    except Exception as error:
        print(f"Crawl4AI not available or failed for {url} ({error}), using requests fallback...")

    # 2. Fallback to requests + BeautifulSoup
    return _extract_via_requests(url)


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        output = DATA_DIR / f"article_{index:02d}.json"
        if output.exists() and output.stat().st_size > 100:
            print(f"Article already exists: {output}")
            continue
        try:
            article = await crawl_article(url)
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")



if __name__ == "__main__":
    asyncio.run(crawl_all())
