#!/usr/bin/env python3
"""
DBQNA Wikipedia + PDF Collector
Collects Wikipedia articles in the ML/NLP/GenAI domain with polite throttling,
resumable manifests, and safe PDF download from a safelist of domains.
"""
#!/usr/bin/env python3
import os, re, json, time, hashlib, logging, random
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm

# ----------------------------- CONFIG ---------------------------------
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "DBQNA-Collector/2.0 (contact: amolchilame49@gmail.com)"
}
DEFAULT_TARGET = 5000
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.jsonl"

PDF_DOMAIN_SAFELIST = {
    "arxiv.org", "openreview.net", "aclanthology.org",
    "jmlr.org", "proceedings.mlr.press", "papers.nips.cc",
    "openaccess.thecvf.com", "osf.io", "ceur-ws.org",
}

SEED_QUERIES = [
    "machine learning", "deep learning", "natural language processing",
    "generative artificial intelligence", "large language model",
    "transformer (machine learning)", "retrieval-augmented generation",
    "semantic search", "vector database", "LangChain", "Mistral (AI)", "LLaMA",
]

CATEGORIES = [
    "Category:Machine learning",
    "Category:Deep learning",
    "Category:Natural language processing",
    "Category:Large language models",
    "Category:Transformers (machine learning)",
]

CONTENT_KEYWORDS = [
    "machine learning","deep learning","natural language processing","nlp",
    "transformer","bert","roberta","albert","xlnet","t5","gpt","llama","mistral","rag",
    "semantic search","faiss","bm25","vector database"
]

EXCLUDE_TITLE_PATTERNS = [
    r"^Help:", r"^Talk:", r"^User:", r"^Wikipedia:", r"^File:", r"^Draft:",
    r"^Portal:", r"^Template:", r"^Category:", r"^Book:", r"^Module:",
    r"^Index:", r"^Outline:"
]

session = requests.Session()
session.headers.update(HEADERS)

# ----------------------------- UTILITIES ------------------------------

def stable_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def slugify(title: str) -> str:
    s = re.sub(r"[^\w\- ]+", "", title).strip().replace(" ", "_")
    return s[:120] or "untitled"

def is_excluded_title(title: str) -> bool:
    return any(re.search(pat, title, flags=re.IGNORECASE) for pat in EXCLUDE_TITLE_PATTERNS)

def polite_sleep(base: float = 0.15, jitter: float = 0.15):
    time.sleep(base + random.random() * jitter)

# ----------------------------- ROBUST API -----------------------------

def api_get(params: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
    params = {**params, "format": "json"}
    for attempt in range(max_retries):
        try:
            r = session.get(API, params=params, timeout=(10, 60))  # connect, read
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt + random.random()
            logging.warning(f"API GET error {e} (attempt {attempt+1}/{max_retries}) – sleeping {wait:.1f}s")
            time.sleep(wait)
    logging.error("Max retries exceeded for Wikipedia API")
    return {}

# ----------------------------- DATA RETRIEVAL -------------------------

def search_pages(query: str, max_results: int = 120) -> List[Dict[str, Any]]:
    results, sroffset = [], 0
    while len(results) < max_results:
        limit = min(50, max_results - len(results))
        data = api_get({
            "action": "query","list": "search","srsearch": query,
            "srnamespace": 0,"srlimit": limit,"sroffset": sroffset
        })
        batch = data.get("query", {}).get("search", [])
        if not batch: break
        results.extend(batch); sroffset += len(batch)
        if "continue" not in data: break
        polite_sleep()
    return results

def category_members(category: str, max_pages: int = 1000) -> List[Dict[str, Any]]:
    members, cmcontinue = [], None
    while len(members) < max_pages:
        params = {
            "action": "query","list": "categorymembers","cmtitle": category,
            "cmnamespace": 0,"cmlimit": 500
        }
        if cmcontinue: params["cmcontinue"] = cmcontinue
        data = api_get(params)
        batch = data.get("query", {}).get("categorymembers", [])
        if not batch: break
        members.extend(batch)
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue: break
        polite_sleep()
    return members[:max_pages]

def get_page_info(pageid: int) -> Optional[Dict[str, Any]]:
    data = api_get({
        "action": "query","pageids": pageid,"prop": "info|pageprops|categories",
        "inprop": "url","cllimit": "max","clshow": "!hidden","formatversion": "2",
    })
    pages = data.get("query", {}).get("pages", [])
    return pages[0] if pages else None

def get_page_html(pageid: int) -> Optional[str]:
    data = api_get({
        "action": "parse","pageid": pageid,"prop": "text","formatversion": "2",
        "disablelimitreport": "1","disableeditsection": "1","disabletoc": "1",
    })
    parse = data.get("parse")
    return parse.get("text") if parse else None

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="mw-parser-output") or soup
    for sel in ["table.infobox","table.navbox","div.thumb","div.reflist","ol.references"]:
        for el in content.select(sel): el.decompose()
    chunks = []
    for el in content.descendants:
        if isinstance(el, NavigableString): continue
        if getattr(el, "name", "") in {"p","h2","h3","h4","li"}:
            t = el.get_text(" ", strip=True)
            if t: chunks.append(t)
    text = "\n".join(chunks)
    return re.sub(r"\n{2,}", "\n\n", text).strip()

def relevant(title: str, categories: List[str], text: str) -> bool:
    title_l = title.lower()
    cats_l = [c.lower() for c in categories]
    text_l = text.lower()
    cat_hit = any(any(k in c for k in ["machine learning","deep learning","language model","neural"]) for c in cats_l)
    hits = {kw for kw in CONTENT_KEYWORDS if kw in title_l or kw in text_l}
    return cat_hit and len(hits) >= 1

def extract_pdf_links(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("//"): href = "https:" + href
        if not href.startswith("http"): continue
        host = urlparse(href).netloc.lower().lstrip("www.")
        if host not in PDF_DOMAIN_SAFELIST: continue
        if href.lower().endswith(".pdf") or "arxiv.org/pdf" in href:
            links.append(href)
    out, seen = [], set()
    for u in links:
        if u not in seen: seen.add(u); out.append(u)
    return out[:5]

def download_pdf(url: str, out_dir: Path, base_id: str, max_mb: int = 25) -> Optional[str]:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{base_id}_{stable_id(url)}.pdf"
        path = out_dir / fname
        if path.exists(): return str(path)
        with session.get(url, stream=True, timeout=(10, 120)) as r:
            r.raise_for_status()
            size = 0
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk: continue
                    f.write(chunk); size += len(chunk)
                    if size > max_mb * 1024 * 1024:
                        f.close(); path.unlink(missing_ok=True); return None
        return str(path)
    except Exception as e:
        logging.warning(f"PDF download failed for {url}: {e}")
        return None

def save_doc(page: Dict[str, Any], text: str, pdf_paths: Optional[List[str]]=None) -> Dict[str, Any]:
    title = page.get("title", "Untitled")
    pageid = page.get("pageid")
    categories = [c.get("title", "Category:") for c in page.get("categories", [])]
    url = page.get("fullurl", f"https://en.wikipedia.org/?curid={pageid}")
    doc_hash = stable_id(f"{pageid}-{title}")
    fname = f"wiki_{doc_hash}_{slugify(title)}.txt"
    path = OUT_DIR / fname
    header = [
        f"Title: {title}",f"URL: {url}",f"PageID: {pageid}",
        f"Categories: {', '.join(categories)}","Source: Wikipedia (CC BY-SA 4.0).","",
        "-----","",
    ]
    path.write_text("\n".join(header) + text + "\n", encoding="utf-8")
    return {
        "doc_id": doc_hash,"title": title,"url": url,"pageid": pageid,
        "categories": categories,"path": str(path),"source": "wikipedia",
        "license": "CC BY-SA 4.0","pdfs": pdf_paths or [],
    }

# ----------------------------- PIPELINE --------------------------------

def collect_candidates(target_count: int) -> List[Dict[str, Any]]:
    seen, cands = set(), []
    for cat in tqdm(CATEGORIES, desc="Crawling categories"):
        mems = category_members(cat, max_pages=1200)
        for m in mems:
            pid, title = m.get("pageid"), m.get("title", "")
            if not pid or not title or is_excluded_title(title): continue
            if pid in seen: continue
            seen.add(pid); cands.append({"pageid": pid, "title": title})
        if len(cands) >= target_count * 3: break
    if len(cands) < target_count * 3:
        for q in tqdm(SEED_QUERIES, desc="Searching seeds"):
            res = search_pages(q, max_results=150)
            for r in res:
                pid, title = r.get("pageid"), r.get("title", "")
                if not pid or not title or is_excluded_title(title): continue
                if pid in seen: continue
                seen.add(pid); cands.append({"pageid": pid, "title": title})
            if len(cands) >= target_count * 4: break
    return cands

def load_existing_manifest_ids(manifest_path: Path) -> set:
    ids = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line); pid = obj.get("pageid")
                    if pid: ids.add(pid)
                except: continue
    return ids

# ----------------------------- MAIN ------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Collect Wikipedia articles for ML/NLP/GenAI domain.")
    ap.add_argument("--n", type=int, default=DEFAULT_TARGET)
    ap.add_argument("--fetch-pdfs", action="store_true")
    ap.add_argument("--pdf-dir", type=str, default="data/pdfs")
    ap.add_argument("--min-chars", type=int, default=800)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(f"Output directory: {OUT_DIR}")
    logging.info(f"Target documents: {args.n}")

    existing_ids = load_existing_manifest_ids(MANIFEST_PATH)
    if existing_ids:
        logging.info(f"Resuming. Found {len(existing_ids)} entries in existing manifest.")

    candidates = collect_candidates(args.n)
    logging.info(f"Collected {len(candidates)} candidate pages.")

    count = len(existing_ids)
    mf = MANIFEST_PATH.open("a", encoding="utf-8")
    pdf_dir = Path(args.pdf_dir)

    try:
        for cand in tqdm(candidates, desc="Processing pages"):
            if count >= args.n: break
            pid = cand["pageid"]
            if pid in existing_ids: continue

            info = get_page_info(pid)
            if not info or info.get("invalid"): continue
            if "disambiguation" in info.get("pageprops", {}): continue
            title = info.get("title", "")
            if is_excluded_title(title): continue
            cats = [c.get("title", "") for c in (info.get("categories") or [])]
            html = get_page_html(pid)
            if not html: continue

            text = html_to_text(html)
            if len(text) < args.min_chars: continue
            if not relevant(title, cats, text): continue

            pdf_paths = []
            if args.fetch_pdfs:
                for u in extract_pdf_links(html):
                    path = download_pdf(u, pdf_dir, base_id=str(pid))
                    if path: pdf_paths.append(path)
                    polite_sleep(0.1,0.1)

            rec = save_doc(info, text, pdf_paths=pdf_paths)
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n"); mf.flush()

            existing_ids.add(pid); count += 1
            polite_sleep()

            if count % 50 == 0: logging.info(f"Wrote {count} documents so far...")

        logging.info(f"Done. Wrote total {count} documents to {OUT_DIR}")
        logging.info(f"Manifest: {MANIFEST_PATH}")
        if count < args.n:
            logging.info("Fewer than target documents were found after filtering. "
                         "Consider adding more seeds or relaxing filters.")
    finally:
        mf.close()

if __name__ == "__main__":
    main()




# import os
# import re
# import json
# import time
# import random
# import hashlib
# import logging
# from pathlib import Path
# from typing import Dict, Any, List, Optional
# from urllib.parse import urlparse

# import requests
# from bs4 import BeautifulSoup, NavigableString
# from tqdm import tqdm
# from requests.adapters import HTTPAdapter, Retry

# # ---------------------------------------
# # Configuration
# # ---------------------------------------

# API = "https://en.wikipedia.org/w/api.php"
# HEADERS = {
#     "User-Agent": "DBQNA-Collector/2.0 (contact: amolchilame49@gmail.com)"
# }

# DEFAULT_TARGET = 5000
# OUT_DIR = Path("data")
# OUT_DIR.mkdir(parents=True, exist_ok=True)
# MANIFEST_PATH = OUT_DIR / "manifest.jsonl"

# PDF_DOMAIN_SAFELIST = {
#     "arxiv.org", "openreview.net", "aclanthology.org", "jmlr.org",
#     "proceedings.mlr.press", "papers.nips.cc", "openaccess.thecvf.com",
#     "osf.io", "ceur-ws.org",
# }

# SEED_QUERIES = [
#     "machine learning", "deep learning", "natural language processing",
#     "generative artificial intelligence", "large language model",
#     "transformer (machine learning)", "self-attention", "attention mechanism",
#     "BERT (language model)", "RoBERTa", "ALBERT (language model)", "XLNet",
#     "T5 (language model)", "GPT-2", "GPT-3", "GPT-4", "LLaMA", "Mistral (AI)",
#     "Qwen (large language model)", "Vision Transformer", "Swin Transformer",
#     "Conformer (neural networks)", "reinforcement learning",
#     "retrieval-augmented generation", "semantic search", "vector database",
# ]

# CATEGORIES = [
#     "Category:Machine learning",
#     "Category:Deep learning",
#     "Category:Artificial neural networks",
#     "Category:Natural language processing",
#     "Category:Computational linguistics",
#     "Category:Large language models",
# ]

# CONTENT_KEYWORDS = [
#     "machine learning", "deep learning", "natural language processing",
#     "transformer", "attention", "bert", "roberta", "llama", "language model",
#     "retrieval-augmented generation", "rag", "semantic search", "vector database",
# ]

# EXCLUDE_TITLE_PATTERNS = [
#     r"^Help:", r"^Talk:", r"^User:", r"^Wikipedia:", r"^File:", r"^Draft:",
#     r"^Portal:", r"^Template:", r"^Category:", r"^Book:", r"^Module:",
#     r"^Index:", r"^Outline:"
# ]

# # ---------------------------------------
# # Session & Utilities
# # ---------------------------------------

# session = requests.Session()
# retries = Retry(total=5, backoff_factor=0.7,
#                 status_forcelist=[429, 500, 502, 503, 504])
# session.mount("https://", HTTPAdapter(max_retries=retries))
# session.headers.update(HEADERS)


# def polite_sleep(base=0.2, jitter=0.3):
#     time.sleep(base + random.random() * jitter)


# def stable_id(s: str) -> str:
#     return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# def slugify(title: str) -> str:
#     s = re.sub(r"[^\w\- ]+", "", title).strip().replace(" ", "_")
#     return s[:120] or "untitled"


# def is_excluded_title(title: str) -> bool:
#     for pat in EXCLUDE_TITLE_PATTERNS:
#         if re.search(pat, title, flags=re.IGNORECASE):
#             return True
#     return False


# def api_get(params: Dict[str, Any]) -> Dict[str, Any]:
#     params = {**params, "format": "json"}
#     r = session.get(API, params=params, timeout=30)
#     r.raise_for_status()
#     return r.json()

# # ---------------------------------------
# # Wikipedia Fetchers
# # ---------------------------------------


# def search_pages(query: str, max_results: int = 120) -> List[Dict[str, Any]]:
#     results = []
#     sroffset = 0
#     while len(results) < max_results:
#         limit = min(50, max_results - len(results))
#         data = api_get({
#             "action": "query",
#             "list": "search",
#             "srsearch": query,
#             "srnamespace": 0,
#             "srlimit": limit,
#             "sroffset": sroffset,
#             "srqiprofile": "classic_noboostlinks",
#         })
#         batch = data.get("query", {}).get("search", [])
#         if not batch:
#             break
#         results.extend(batch)
#         sroffset += len(batch)
#         if "continue" not in data:
#             break
#         polite_sleep()
#     return results


# def category_members(category: str, max_pages: int = 1000) -> List[Dict[str, Any]]:
#     members = []
#     cmcontinue = None
#     while len(members) < max_pages:
#         params = {
#             "action": "query",
#             "list": "categorymembers",
#             "cmtitle": category,
#             "cmnamespace": 0,
#             "cmlimit": 500,
#         }
#         if cmcontinue:
#             params["cmcontinue"] = cmcontinue
#         data = api_get(params)
#         batch = data.get("query", {}).get("categorymembers", [])
#         if not batch:
#             break
#         members.extend(batch)
#         cmcontinue = data.get("continue", {}).get("cmcontinue")
#         if not cmcontinue:
#             break
#         polite_sleep()
#     return members[:max_pages]


# def get_page_info(pageid: int) -> Optional[Dict[str, Any]]:
#     data = api_get({
#         "action": "query",
#         "pageids": pageid,
#         "prop": "info|pageprops|categories",
#         "inprop": "url",
#         "cllimit": "max",
#         "clshow": "!hidden",
#         "formatversion": "2",
#     })
#     pages = data.get("query", {}).get("pages", [])
#     return pages[0] if pages else None


# def get_page_html(pageid: int) -> Optional[str]:
#     data = api_get({
#         "action": "parse",
#         "pageid": pageid,
#         "prop": "text",
#         "formatversion": "2",
#         "disablelimitreport": "1",
#         "disableeditsection": "1",
#         "disabletoc": "1",
#     })
#     parse = data.get("parse")
#     return parse.get("text") if parse else None

# # ---------------------------------------
# # Processing
# # ---------------------------------------


# def html_to_text(html: str) -> str:
#     soup = BeautifulSoup(html, "html.parser")
#     content = soup.find("div", class_="mw-parser-output") or soup
#     for sel in [
#         "table.infobox", "table.vertical-navbox", "table.navbox",
#         "table.metadata", "div.thumb", "div.reflist", "ol.references",
#         "sup.reference", "span.mw-editsection", "div.hatnote",
#         "div#toc", "span.coordinates"
#     ]:
#         for el in content.select(sel):
#             el.decompose()
#     chunks = []
#     for el in content.descendants:
#         name = getattr(el, "name", "")
#         if name in {"p", "h2", "h3", "h4", "li"}:
#             t = el.get_text(" ", strip=True)
#             if t:
#                 chunks.append(t)
#     text = "\n".join(chunks)
#     text = re.sub(r"\n{2,}", "\n\n", text)
#     return text.strip()


# def relevant(title: str, categories: List[str], text: str) -> bool:
#     title_l = title.lower()
#     cats_l = [c.lower() for c in categories]
#     text_l = text.lower()
#     cat_hit = any(
#         any(key in c for key in
#             ["machine learning", "deep learning", "natural language processing",
#              "artificial intelligence", "neural network", "language model",
#              "transformer", "reinforcement learning", "information retrieval"])
#         for c in cats_l
#     )
#     hits = any(kw.lower() in title_l or kw.lower() in text_l
#                for kw in CONTENT_KEYWORDS)
#     return cat_hit and hits


# def extract_pdf_links(html: str) -> List[str]:
#     soup = BeautifulSoup(html, "html.parser")
#     links = []
#     for a in soup.find_all("a", href=True):
#         href = a["href"]
#         if href.startswith("//"):
#             href = "https:" + href
#         if not href.startswith("http"):
#             continue
#         host = urlparse(href).netloc.lower()
#         if host.startswith("www."):
#             host = host[4:]
#         if host not in PDF_DOMAIN_SAFELIST:
#             continue
#         if href.lower().endswith(".pdf") or "arxiv.org/pdf" in href or "openreview.net/pdf" in href:
#             links.append(href)
#     seen = set()
#     out = []
#     for u in links:
#         if u not in seen:
#             seen.add(u)
#             out.append(u)
#     return out[:5]


# def download_pdf(url: str, out_dir: Path, base_id: str, max_mb: int = 25) -> Optional[str]:
#     try:
#         out_dir.mkdir(parents=True, exist_ok=True)
#         fname = f"{base_id}_{stable_id(url)}.pdf"
#         path = out_dir / fname
#         if path.exists():
#             return str(path)
#         with session.get(url, stream=True, timeout=60) as r:
#             r.raise_for_status()
#             if "application/pdf" not in r.headers.get("Content-Type", ""):
#                 return None
#             size = 0
#             with open(path, "wb") as f:
#                 for chunk in r.iter_content(chunk_size=8192):
#                     if not chunk:
#                         continue
#                     f.write(chunk)
#                     size += len(chunk)
#                     if size > max_mb * 1024 * 1024:
#                         f.close()
#                         path.unlink(missing_ok=True)
#                         return None
#         return str(path)
#     except Exception as e:
#         logging.error(f"PDF download failed for {url}: {e}")
#         return None


# def save_doc(page: Dict[str, Any], text: str, pdf_paths: Optional[List[str]] = None) -> Dict[str, Any]:
#     title = page.get("title", "Untitled")
#     pageid = page.get("pageid")
#     categories = [c.get("title", "Category:") for c in page.get("categories", [])]
#     url = page.get("fullurl", f"https://en.wikipedia.org/?curid={pageid}")
#     doc_hash = stable_id(f"{pageid}-{title}")
#     fname = f"wiki_{doc_hash}_{slugify(title)}.txt"
#     path = OUT_DIR / fname
#     header = [
#         f"Title: {title}",
#         f"URL: {url}",
#         f"PageID: {pageid}",
#         f"Categories: {', '.join(categories)}",
#         "Source: Wikipedia (CC BY-SA 4.0). Content may require attribution.",
#         "",
#         "-----",
#         "",
#     ]
#     path.write_text("\n".join(header) + text + "\n", encoding="utf-8")
#     rec = {
#         "doc_id": doc_hash,
#         "title": title,
#         "url": url,
#         "pageid": pageid,
#         "categories": categories,
#         "path": str(path),
#         "source": "wikipedia",
#         "license": "CC BY-SA 4.0",
#         "pdfs": pdf_paths or [],
#     }
#     return rec


# def collect_candidates(target_count: int) -> List[Dict[str, Any]]:
#     seen = set()
#     cands: List[Dict[str, Any]] = []

#     for cat in tqdm(CATEGORIES, desc="Crawling categories"):
#         mems = category_members(cat, max_pages=1200)
#         for m in mems:
#             pid = m.get("pageid")
#             title = m.get("title", "")
#             if not pid or not title or is_excluded_title(title):
#                 continue
#             if pid in seen:
#                 continue
#             seen.add(pid)
#             cands.append({"pageid": pid, "title": title})
#         if len(cands) >= target_count * 3:
#             break
#         polite_sleep()

#     if len(cands) < target_count * 3:
#         for q in tqdm(SEED_QUERIES, desc="Searching seeds"):
#             res = search_pages(q, max_results=150)
#             for r in res:
#                 pid = r.get("pageid")
#                 title = r.get("title", "")
#                 if not pid or not title or is_excluded_title(title):
#                     continue
#                 if pid in seen:
#                     continue
#                 seen.add(pid)
#                 cands.append({"pageid": pid, "title": title})
#             if len(cands) >= target_count * 4:
#                 break
#             polite_sleep()

#     return cands


# def load_existing_manifest_ids(manifest_path: Path) -> set:
#     ids = set()
#     if manifest_path.exists():
#         with manifest_path.open("r", encoding="utf-8") as f:
#             for line in f:
#                 try:
#                     obj = json.loads(line)
#                     pid = obj.get("pageid")
#                     if pid:
#                         ids.add(pid)
#                 except Exception:
#                     continue
#     return ids

# # ---------------------------------------
# # Main
# # ---------------------------------------


# def main():
#     import argparse
#     ap = argparse.ArgumentParser(description="Collect Wikipedia articles for ML/NLP/GenAI domain.")
#     ap.add_argument("--n", type=int, default=DEFAULT_TARGET, help="Target number of documents (e.g., 500 or 5000)")
#     ap.add_argument("--fetch-pdfs", action="store_true", help="Also fetch open-access PDFs from a safe domain list")
#     ap.add_argument("--pdf-dir", type=str, default="data/pdfs", help="Folder to store PDFs")
#     ap.add_argument("--min-chars", type=int, default=800, help="Skip articles with fewer characters")
#     args = ap.parse_args()

#     logging.basicConfig(level=logging.INFO, format="%(message)s")
#     logging.info(f"Output directory: {OUT_DIR}")
#     logging.info(f"Target documents: {args.n}")

#     existing_ids = load_existing_manifest_ids(MANIFEST_PATH)
#     if existing_ids:
#         logging.info(f"Resuming. Found {len(existing_ids)} entries in existing manifest.")

#     candidates = collect_candidates(args.n)
#     logging.info(f"Collected {len(candidates)} candidate pages.")

#     count = len(existing_ids)

#     pdf_dir = Path(args.pdf_dir)

#     for cand in tqdm(candidates, desc="Processing pages"):
#         if count >= args.n:
#             break
#         pid = cand["pageid"]
#         if pid in existing_ids:
#             continue

#         info = get_page_info(pid)
#         if not info or info.get("invalid"):
#             continue
#         if "disambiguation" in info.get("pageprops", {}):
#             continue

#         title = info.get("title", "")
#         if is_excluded_title(title):
#             continue

#         cats = [c.get("title", "") for c in (info.get("categories") or [])]
#         html = get_page_html(pid)
#         if not html:
#             continue

#         text = html_to_text(html)
#         if len(text) < args.min_chars:
#             continue

#         if not relevant(title, cats, text):
#             continue

#         pdf_paths = []
#         if args.fetch_pdfs:
#             links = extract_pdf_links(html)
#             for u in links:
#                 path = download_pdf(u, pdf_dir, base_id=str(pid))
#                 if path:
#                     pdf_paths.append(path)
#                 polite_sleep()

#         rec = save_doc(info, text, pdf_paths=pdf_paths)
#         with MANIFEST_PATH.open("a", encoding="utf-8") as mf:
#             mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

#         existing_ids.add(pid)
#         count += 1

#         polite_sleep()

#         if count % 50 == 0:
#             logging.info(f"Wrote {count} documents so far...")

#     logging.info(f"Done. Wrote total {count} documents to {OUT_DIR}")
#     logging.info(f"Manifest: {MANIFEST_PATH}")
#     if count < args.n:
#         logging.info("Fewer than target documents were found after filtering. "
#                      "Consider adding more seeds or relaxing filters.")


# if __name__ == "__main__":
#     main()
