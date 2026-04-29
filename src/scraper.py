"""
scraper.py  —  Web scraper for the UChicago MS in Applied Data Science website.
Traverses the main page and all linked sub-pages, cleaning and structuring the
text for downstream RAG processing.
"""

import time
import re
from urllib.parse import urljoin, urlparse
from typing import List, Dict

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/"

SUBPAGES = [
    "course-progressions/",
    "in-person-program/",
    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/faqs/",
    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/career-outcomes/",
    "https://datascience.uchicago.edu/education/tuition-fees-aid/",
    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/how-to-apply/",
    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/events-deadlines/",
    "https://applieddatascience.psd.uchicago.edu/academics/curriculum/",
    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/instructors-staff/",
    "https://datascience.uchicago.edu/education/masters-programs/online-program/",
    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/capstone-projects/",
    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/capstone-project-archive/",
    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/our-students/",
    "https://datascience.uchicago.edu/explore-the-ms-ads-campus/",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAG-research-bot/1.0)"
}


def clean_text(raw: str) -> str:
    """Remove extra whitespace and non-printable characters."""
    text = re.sub(r"\s+", " ", raw)
    text = text.strip()
    return text


def scrape_page(url: str) -> Dict[str, str]:
    """
    Fetch a single page and extract the title + main body text.
    Returns a dict with keys: url, title, content.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[WARN] Could not fetch {url}: {exc}")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Title
    tag = soup.find("title")
    title = clean_text(tag.get_text()) if tag else url

    # Remove nav/footer noise
    for tag in soup.find_all(["nav", "footer", "header", "script", "style"]):
        tag.decompose()

    # Main content area — try common selectors
    for selector in ["main", "article", ".entry-content", "#content", "body"]:
        main = soup.select_one(selector)
        if main:
            break

    content = clean_text(main.get_text(separator=" ")) if main else ""
    return {"url": url, "title": title, "content": content}


def scrape_all() -> List[Dict[str, str]]:
    """Scrape the main page and all known sub-pages."""
    urls = [BASE_URL] + [
        u if u.startswith("http") else urljoin(BASE_URL, u) for u in SUBPAGES
    ]
    documents = []
    for url in urls:
        print(f"Scraping: {url}")
        doc = scrape_page(url)
        if doc:
            documents.append(doc)
        time.sleep(0.5)   # polite crawling delay
    print(f"\n✅  Scraped {len(documents)} pages.")
    return documents


if __name__ == "__main__":
    import json, pathlib
    docs = scrape_all()
    out = pathlib.Path("data/raw_pages.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(docs, indent=2))
    print(f"Saved to {out}")
