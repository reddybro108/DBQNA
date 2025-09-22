#!/usr/bin/env python3
"""
DBQNA Wikipedia Collector – Enterprise-Grade Refactor
"""
import os
import re
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm

class WikipediaCollector:
    API = "https://en.wikipedia.org/w/api.php"
    HEADERS = {"User-Agent": "DBQNA-Collector/2.0 (contact: amolchilame49@gmail.com)"}

    def __init__(
        self,
        out_dir: str = "data",
        target_docs: int = 500,
        seed_queries: Optional[List[str]] = None,
        content_keywords: Optional[List[str]] = None,
        category_keys: Optional[List[str]] = None,
    ):
        self.target_docs = target_docs
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_dir / "manifest.jsonl"

        # Inject your project’s seed queries or default to a small set
        self.seed_queries = seed_queries or ["machine learning", "natural language processing"]
        self.content_keywords = content_keywords or ["machine learning", "transformer", "language model"]
        self.category_keys = category_keys or ["machine learning", "artificial intelligence"]

        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

        self.exclude_title_patterns = [
            r"^Help:", r"^Talk:", r"^User:", r"^Wikipedia:", r"^File:", r"^Draft:",
            r"^Portal:", r"^Template:", r"^Category:", r"^Book:", r"^Module:",
            r"^Index:", r"^Outline:"
        ]

    # ---------- Utility methods ----------
    @staticmethod
    def stable_id(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def slugify(title: str) -> str:
        s = re.sub(r"[^\w\- ]+", "", title).strip().replace(" ", "_")
        return s[:120] or "untitled"

    def is_excluded_title(self, title: str) -> bool:
        for pat in self.exclude_title_patterns:
            if re.search(pat, title, flags=re.IGNORECASE):
                return True
        return False

    def api_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = {**params, "format": "json"}
        for _ in range(3):
            try:
                r = self.session.get(self.API, params=params, timeout=30)
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                time.sleep(0.5)
        raise RuntimeError("Failed API request after retries")

    # ---------- Wikipedia API wrappers ----------
    def search_pages(self, query: str, max_results: int = 60) -> List[Dict[str, Any]]:
        results = []
        sroffset = 0
        while len(results) < max_results:
            limit = min(50, max_results - len(results))
            data = self.api_get({
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srnamespace": 0,
                "srlimit": limit,
                "sroffset": sroffset,
                "srqiprofile": "classic_noboostlinks",
            })
            batch = data.get("query", {}).get("search", [])
            if not batch:
                break
            results.extend(batch)
            sroffset += len(batch)
            if "continue" not in data:
                break
            time.sleep(0.1)
        return results

    def get_page_info(self, pageid: int) -> Optional[Dict[str, Any]]:
        data = self.api_get({
            "action": "query",
            "pageids": pageid,
            "prop": "info|pageprops|categories",
            "inprop": "url",
            "cllimit": "max",
            "clshow": "!hidden",
            "formatversion": "2",
        })
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return None
        return pages[0]

    def get_page_html(self, pageid: int) -> Optional[str]:
        data = self.api_get({
            "action": "parse",
            "pageid": pageid,
            "prop": "text",
            "formatversion": "2",
            "disablelimitreport": "1",
            "disableeditsection": "1",
            "disabletoc": "1",
        })
        parse = data.get("parse")
        if not parse:
            return None
        return parse.get("text")

    @staticmethod
    def html_to_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        content = soup.find("div", class_="mw-parser-output") or soup
        for sel in [
            "table.infobox", "table.vertical-navbox", "table.navbox", "table.metadata",
            "div.thumb", "div.reflist", "ol.references", "sup.reference", "span.mw-editsection",
            "div.hatnote", "div#toc", "span.coordinates"
        ]:
            for el in content.select(sel):
                el.decompose()
        text_lines = []
        for el in content.descendants:
            if isinstance(el, NavigableString):
                continue
            name = getattr(el, "name", "")
            if name in {"p", "h2", "h3", "h4", "li"}:
                chunk = el.get_text(" ", strip=True)
                if chunk:
                    text_lines.append(chunk)
        text = "\n".join(text_lines)
        text = re.sub(r"\n{2,}", "\n\n", text)
        return text.strip()

    def relevant(self, title: str, categories: List[str], text: str) -> bool:
        title_l = title.lower()
        cats_l = [c.lower() for c in categories]
        text_l = text.lower()

        cat_hit = any(any(ck in c for ck in self.category_keys) for c in cats_l)
        hits = set()
        for kw in self.content_keywords:
            kw_l = kw.lower()
            if kw_l in title_l or kw_l in text_l:
                hits.add(kw_l)
        return cat_hit and (len(hits) >= 1)

    def save_doc(self, page: Dict[str, Any], text: str) -> Dict[str, Any]:
        title = page.get("title", "Untitled")
        pageid = page.get("pageid")
        url = page.get("fullurl", f"https://en.wikipedia.org/?curid={pageid}")
        categories = [c.get("title", "Category:") for c in page.get("categories", [])]
        doc_hash = self.stable_id(f"{pageid}-{title}")
        fname = f"wiki_{doc_hash}_{self.slugify(title)}.txt"
        path = self.out_dir / fname

        header = [
            f"Title: {title}",
            f"URL: {url}",
            f"PageID: {pageid}",
            f"Categories: {', '.join(categories)}",
            "Source: Wikipedia (CC BY-SA 4.0). Content may require attribution.",
            "",
            "-----",
            "",
        ]
        path.write_text("\n".join(header) + text + "\n", encoding="utf-8")

        rec = {
            "doc_id": doc_hash,
            "title": title,
            "url": url,
            "pageid": pageid,
            "categories": categories,
            "path": str(path),
            "source": "wikipedia",
            "license": "CC BY-SA 4.0",
        }
        return rec

    def collect_candidates(self) -> List[Dict[str, Any]]:
        seen_ids = set()
        candidates: List[Dict[str, Any]] = []
        for q in tqdm(self.seed_queries, desc="Searching seeds"):
            results = self.search_pages(q, max_results=80)
            for r in results:
                pid = r.get("pageid")
                title = r.get("title", "")
                if not pid or not title or self.is_excluded_title(title):
                    continue
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                candidates.append({"pageid": pid, "title": title})
            if len(seen_ids) >= self.target_docs * 3:
                break
            time.sleep(0.1)
        return candidates

    def run(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.info(f"Starting Wikipedia collection to {self.out_dir} (target {self.target_docs})")

        candidates = self.collect_candidates()
        logging.info(f"Collected {len(candidates)} candidate pages from search.")

        count = 0
        written_ids = set()
        manifest_f = self.manifest_path.open("w", encoding="utf-8")

        try:
            for cand in tqdm(candidates, desc="Processing pages"):
                if count >= self.target_docs:
                    break
                pid = cand["pageid"]
                if pid in written_ids:
                    continue

                info = self.get_page_info(pid)
                if not info or info.get("invalid"):
                    continue

                pageprops = info.get("pageprops", {})
                if "disambiguation" in pageprops:
                    continue

                title = info.get("title", "")
                if self.is_excluded_title(title):
                    continue

                cats = [c.get("title", "") for c in (info.get("categories") or [])]
                html = self.get_page_html(pid)
                if not html:
                    continue

                text = self.html_to_text(html)
                if len(text) < 600:
                    continue

                if not self.relevant(title, cats, text):
                    continue

                rec = self.save_doc(info, text)
                manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                manifest_f.flush()

                written_ids.add(pid)
                count += 1

                time.sleep(0.1)
                if count % 50 == 0:
                    logging.info(f"Wrote {count} documents...")

            logging.info(f"Done. Wrote {count} documents to {self.out_dir}")
            logging.info(f"Manifest: {self.manifest_path}")
        finally:
            manifest_f.close()

if __name__ == "__main__":
    collector = WikipediaCollector(
        out_dir=os.getenv("WIKI_OUT_DIR", "data"),
        target_docs=int(os.getenv("WIKI_TARGET_DOCS", 500)),
    )
    collector.run()
