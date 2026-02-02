# ============================
# UKSC Case -> Judge Assignment (Monolithic Colab Script)
# ============================

import os
import re
import json
import time
import math
import random
import hashlib
import subprocess
import zipfile
from pathlib import Path
import shutil
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from dateutil import parser as dateparser
from sklearn.model_selection import train_test_split

import torch
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT


# ============================
# Config
# ============================

COURT_MODE = "highcourt"  # "supremecourt" | "highcourt"
COURT_MODE_CHOICES = {"supremecourt", "highcourt"}
if COURT_MODE not in COURT_MODE_CHOICES:
    raise ValueError(f"Unsupported COURT_MODE={COURT_MODE}. Choose from {sorted(COURT_MODE_CHOICES)}.")
USE_OPINION_AUTHOR_WEIGHTING = True   # prefer authored opinions when building judge profiles
OPINION_AUTHOR_WEIGHT = 2.0           # weight multiplier when judge authored the opinion
PANEL_JUDGES_ONLY = True              # when True, require <panel> tags for High Court; else fall back to heuristics
CASE_PARSE_VERSION = 4                # bump to invalidate stale per-case cached JSONs (judge parsing etc.)
ZIP_RESULTS = True                    # zip key outputs at end of run for easy download

MAX_CASES = 200               # crawl until we have this many cases with Judgment date
MAX_LIST_PAGES = 200            # safety stop
REQUEST_DELAY_SEC = 0.2         # politeness
HTTP_TIMEOUT = 30
RANDOM_SEED = 42

TEST_SIZE = 0.2
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # strong general embedding model
EMBED_BATCH_SIZE = 64
CHUNK_WORDS = 450
CHUNK_OVERLAP = 60

# Scoring weights (tweakable)
W_SIM = 0.70
W_SPEED = 0.20
W_COMPLEXITY_FIT = 0.10

# if case is urgent, speed matters more
URGENT_SPEED_BOOST = 0.25  # added to W_SPEED * urgency

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; UKSCJudgeAssigner/0.1; +https://www.supremecourt.uk/)"
}
BAILII_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
    )
}

SUPREME_LIST_URL = "https://www.supremecourt.uk/cases"
SUPREME_BASE_URL = "https://www.supremecourt.uk"
SITEMAP_URL_SUPREME = "https://www.supremecourt.uk/sitemap.xml"

HIGHCOURT_LIST_BASE = "https://www.bailii.org/ew/cases/EWHC/KB/"
HIGHCOURT_TOC_TEMPLATE = HIGHCOURT_LIST_BASE + "toc-{}.html"
HIGHCOURT_BASE_URL = "https://www.bailii.org"

if COURT_MODE == "supremecourt":
    BASE_CASES_LIST_URL = SUPREME_LIST_URL
    BASE_CASE_URL = SUPREME_BASE_URL
    CASE_LISTING_MODE = "auto"  # "listing", "sitemap", "auto"
    SITEMAP_URL = SITEMAP_URL_SUPREME
    CACHE_DIR = "/content/uksc_cache"
    CASE_SOURCE_NAME = "UK Supreme Court"
else:
    BASE_CASES_LIST_URL = HIGHCOURT_LIST_BASE
    BASE_CASE_URL = HIGHCOURT_BASE_URL
    CASE_LISTING_MODE = "listing"  # placeholder; not used for BAILII
    SITEMAP_URL = None
    CACHE_DIR = "/content/uksc_cache_highcourt"
    CASE_SOURCE_NAME = "EWHC King's Bench"

os.makedirs(CACHE_DIR, exist_ok=True)
EXPORT_DIR = os.path.join(CACHE_DIR, "export")
os.makedirs(EXPORT_DIR, exist_ok=True)
USE_CACHED_CASES = True
CASES_CACHE_PATH = os.path.join(CACHE_DIR, "cases_raw.json")
CASE_JSON_DIR = os.path.join(CACHE_DIR, "case_json")
CACHE_SAVE_EVERY = 20
CRAWL_DEBUG = True
CRAWL_DEBUG_SAMPLE_IDS = 5
USE_DERIVED_CACHE = True
OVERWRITE_DERIVED_CACHE = False
DERIVED_CACHE_DIR = os.path.join(CACHE_DIR, "derived")
DERIVED_META_PATH = os.path.join(DERIVED_CACHE_DIR, "derived_meta.json")
DERIVED_ARRAYS_PATH = os.path.join(DERIVED_CACHE_DIR, "derived_arrays.npz")
DERIVED_CACHE_VERSION = 5
TEST_REPORT_PATH = os.path.join(EXPORT_DIR, "test_report.json")
KEYWORD_DF_MAX_RATIO = 0.1
KEYWORD_MIN_CHARS = 4
KEYWORD_TOP_CASE_TERMS = 8
KEYWORD_TOP_JUDGE_TERMS = 30
KEYWORD_SIM_THRESHOLD = 0.55
KEYWORD_SIM_TOP_N = 8

GENERIC_KW_STOPWORDS = {
    "court", "high", "bench", "division", "king", "queen", "england", "wales",
    "royal", "justice", "judgment", "judgement", "appeal", "appellate"
}
GENERIC_KW_PATTERN = re.compile(r"\b(court|high|bench|division|king|queen|england|wales|royal)\b", re.IGNORECASE)

os.makedirs(DERIVED_CACHE_DIR, exist_ok=True)
os.makedirs(CASE_JSON_DIR, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================
# Utilities
# ============================

def _safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def clean_text(s: str) -> str:
    s = re.sub(r"\r", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def parse_date_maybe(s: str) -> Optional[str]:
    """
    Returns ISO date (YYYY-MM-DD) or None.
    """
    try:
        dt = dateparser.parse(s, dayfirst=True, fuzzy=True)
        if dt is None:
            return None
        return dt.date().isoformat()
    except Exception:
        return None

def days_between(d1_iso: Optional[str], d2_iso: Optional[str]) -> Optional[int]:
    if not d1_iso or not d2_iso:
        return None
    try:
        d1 = dateparser.parse(d1_iso).date()
        d2 = dateparser.parse(d2_iso).date()
        return int((d2 - d1).days)
    except Exception:
        return None

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)

def request_get(
    url: str,
    session: requests.Session,
    max_retries: int = 3,
    headers: Optional[Dict[str, str]] = None
) -> Optional[requests.Response]:
    req_headers = headers or DEFAULT_HEADERS
    for attempt in range(max_retries):
        try:
            r = session.get(url, headers=req_headers, timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                return r
            # retry on transient-ish failures
            if r.status_code in {429, 500, 502, 503, 504}:
                time.sleep(0.8 * (attempt + 1))
                continue
            return r  # non-200, non-retry
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    return None

def debug_log(msg: str) -> None:
    if CRAWL_DEBUG:
        print(msg)

def print_block(title: str) -> None:
    bar = "=" * 90
    print(f"\n{bar}\n{title}\n{bar}")

def load_cases_cache(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            # Invalidate if parse version missing/mismatch
            if not data:
                return data
            first = data[0] or {}
            if first.get("_parse_version") != CASE_PARSE_VERSION:
                debug_log(f"[cache] cases cache parse_version mismatch; expected {CASE_PARSE_VERSION}")
                return []
            return data
        if isinstance(data, dict) and isinstance(data.get("cases"), list):
            return data["cases"]
    except Exception as exc:
        debug_log(f"[cache] Failed to load cache at {path}: {exc}")
    return []

def save_cases_cache(path: str, cases: List[Dict[str, Any]]) -> None:
    if not path:
        return
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

def sync_export_files():
    """
    Ensure key files are present under EXPORT_DIR for sharing/zipping.
    """
    os.makedirs(EXPORT_DIR, exist_ok=True)
    # cases_raw.json
    if os.path.exists(CASES_CACHE_PATH):
        shutil.copy2(CASES_CACHE_PATH, os.path.join(EXPORT_DIR, "cases_raw.json"))
    # report.html (if present in cache root)
    report_src = os.path.join(CACHE_DIR, "report.html")
    if os.path.exists(report_src):
        shutil.copy2(report_src, os.path.join(EXPORT_DIR, "report.html"))

def zip_outputs(cache_dir: str, zip_results: bool = True) -> Optional[str]:
    """
    Bundle outputs into a zip for easy download (Colab-friendly).
    Returns zip path or None.
    """
    if not zip_results:
        return None
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None

    zip_path = Path("/content") / f"{cache_path.name}.zip"
    try:
        if zip_path.exists():
            zip_path.unlink()
        # Zip only the export folder (outputs) to avoid raw caches
        export_dir = cache_path / "export"
        if export_dir.exists():
            subprocess.check_call(["zip", "-r", str(zip_path), str(export_dir)])
        else:
            # fallback: zip selected output files at cache root
            candidates = [
                "cases_raw.json",
                "test_assignments.json",
                "test_report.json",
                "judge_profiles_summary.json",
                "report.html",
            ]
            args = ["zip", "-r", str(zip_path)] + [str(cache_path / c) for c in candidates if (cache_path / c).exists()]
            subprocess.check_call(args)
    except Exception:
        # Fallback: zip key files only
        to_include = list((cache_path / "export").rglob("*")) if (cache_path / "export").exists() else []
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for src in to_include:
                if src.exists():
                    zf.write(src, arcname=str(src.relative_to(cache_path)))

    # Attempt Colab download
    try:
        from google.colab import files  # type: ignore
        files.download(str(zip_path))
    except Exception:
        pass
    return str(zip_path)

def extract_init_cases_pagination(html: str) -> Optional[Dict[str, int]]:
    prefix = 'initCasesData\\":{\\"pagination\\":{\\"page\\":'
    pattern = re.escape(prefix) + r'(\d+),\\"pageSize\\":(\d+),\\"pageCount\\":(\d+),\\"total\\":(\d+)'
    m = re.search(pattern, html)
    if not m:
        return None
    return {
        "page": int(m.group(1)),
        "page_size": int(m.group(2)),
        "page_count": int(m.group(3)),
        "total": int(m.group(4)),
    }

def extract_init_case_ids(html: str) -> List[str]:
    return re.findall(r'caseId\\\\\":\\\\\"(UKSC/[^\\\\\\\"]+)', html)

def soup_main_text(soup: BeautifulSoup) -> str:
    main = soup.find("main")
    node = main if main else soup
    return clean_text(node.get_text("\n"))

def extract_lines(soup: BeautifulSoup) -> List[str]:
    lines = [x.strip() for x in soup.get_text("\n").splitlines()]
    return [x for x in lines if x and x != "•"]

def normalize_person_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def is_noise_keyword(s: str) -> bool:
    nk = normalize_keyword(s)
    if not nk:
        return True
    if len(nk) < KEYWORD_MIN_CHARS:
        return True
    if GENERIC_KW_PATTERN.search(nk):
        return True
    if any(tok in nk.split() for tok in GENERIC_KW_STOPWORDS):
        return True
    return False

def canonicalize_judge_name(name: str) -> str:
    """
    Normalize judge strings so that variants like
    'HHJ Lickley KC sitting as a Judge of the High Court'
    and 'HHJ LICKLEY KC SITTING AS A' collapse.
    """
    n = (name or "").strip()
    # remove common role phrases
    n = re.sub(r"\s*sitting as.*", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\s*acting as.*", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\s*the honourable\s+", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\s*\(.*?\)\s*$", "", n)
    n = re.sub(r"\s{2,}", " ", n)
    return n.strip().lower()

def dedup_judges_ordered(judges: List[str]) -> List[str]:
    """
    Deduplicate judge names using canonical form and fuzzy similarity.
    """
    from difflib import SequenceMatcher
    seen = []
    out = []
    for j in judges:
        canon = canonicalize_judge_name(j)
        if not canon:
            continue
        duplicate = False
        for c in seen:
            if c == canon:
                duplicate = True
                break
            if SequenceMatcher(None, c, canon).ratio() > 0.9:
                duplicate = True
                break
        if duplicate:
            continue
        seen.append(canon)
        out.append(j)
    return out

def infer_opinion_author(lines: List[str]) -> Optional[str]:
    """
    Heuristic: look for early lines like
      "LORD REED: (with whom ... agree)" or "Judgment given by Lord Reed"
    """
    patterns = [
        r"^(?P<name>(?:lord|lady|mr|mrs|ms|miss|sir|dame|hhj|his honour judge|her honour judge|the hon\.?|mr justice|mrs justice|ms justice|miss justice|lady justice|lord justice)[^:]{0,80}):",
        r"judgment\s+(?:given|delivered)\s+by\s+(?P<name>[^,.;]{3,80})",
        r"^(?P<name>lord\s+[A-Z][A-Za-z\-\s']{1,60})\s*\(with whom"
    ]
    # scan first ~200 lines where headings live
    for ln in lines[:200]:
        text = ln.strip()
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                name = m.group("name").strip()
                name = re.sub(r"\s*\(with whom.*", "", name, flags=re.IGNORECASE)
                name = re.sub(r"\s*[:,–-]+\s*$", "", name)
                name = re.sub(r"\s{2,}", " ", name)
                if name:
                    return name
    # fallback: first standalone judge-style line
    for ln in lines[:120]:
        text = ln.strip()
        if re.match(r"^(the hon\.?\s+)?(lord|lady|mr|mrs|ms|miss|sir|dame)\s+.*justice", text, flags=re.IGNORECASE):
            return re.sub(r"\s{2,}", " ", text)
    return None
def find_value_after_label(lines: List[str], label: str) -> Optional[str]:
    for i in range(len(lines) - 1):
        if lines[i].strip().lower() == label.strip().lower():
            return lines[i + 1].strip()
    return None

def find_block_after_label(lines: List[str], label: str, stop_labels: List[str], max_lines: int = 60) -> Optional[str]:
    """
    Finds a paragraph-ish block after a label until we hit another stop label or exceed max_lines.
    """
    label_low = label.lower()
    stop_set = {s.lower() for s in stop_labels}
    for i in range(len(lines)):
        if lines[i].strip().lower() == label_low:
            out = []
            for j in range(i + 1, min(len(lines), i + 1 + max_lines)):
                if lines[j].strip().lower() in stop_set:
                    break
                out.append(lines[j])
            txt = " ".join(out).strip()
            return txt if txt else None
    return None

def chunk_words(text: str, chunk_words: int = 450, overlap: int = 60) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_words - overlap)
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks

def estimate_case_urgency(text: str) -> float:
    """
    Heuristic urgency score in [0,1] based on presence of typically time-sensitive subject matter.
    Adjust for your own domain assumptions.
    """
    t = (text or "").lower()
    triggers = [
        "bail", "extradition", "detention", "deport", "removal", "immigration detention",
        "child", "children", "adoption", "injunction", "interim", "asylum",
        "custody", "urgent", "habeas", "prison", "probation"
    ]
    hits = sum(1 for w in triggers if w in t)
    return float(min(1.0, hits / 4.0))

def normalize_keyword(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def is_repetitive_keyword(s: str) -> bool:
    parts = s.split()
    return len(parts) > 1 and len(set(parts)) == 1

def compute_cases_hash(cases: List[Dict[str, Any]]) -> str:
    urls = [c.get("case_url", "") or "" for c in cases]
    payload = [str(CASE_PARSE_VERSION)] + urls
    return sha1("\n".join(payload))

def compute_keyword_idf(case_keywords: List[List[str]]) -> Dict[str, float]:
    n_cases = len(case_keywords)
    if n_cases == 0:
        return {}
    df = Counter()
    for kws in case_keywords:
        uniq = set()
        for kw in kws or []:
            nk = normalize_keyword(kw)
            if not nk or len(nk) < KEYWORD_MIN_CHARS:
                continue
            if is_noise_keyword(nk):
                continue
            if is_repetitive_keyword(nk):
                continue
            uniq.add(nk)
        for nk in uniq:
            df[nk] += 1
    out = {}
    for kw, count in df.items():
        if (count / n_cases) > KEYWORD_DF_MAX_RATIO:
            continue
        out[kw] = math.log((1.0 + n_cases) / (1.0 + count)) + 1.0
    return out

def select_top_case_terms(case_keywords: List[str], keyword_idf: Dict[str, float], top_n: int) -> List[str]:
    scored = []
    seen = set()
    for kw in case_keywords or []:
        nk = normalize_keyword(kw)
        if not nk or nk in seen:
            continue
        if nk not in keyword_idf:
            continue
        scored.append((keyword_idf[nk], nk))
        seen.add(nk)
    scored.sort(reverse=True)
    return [k for _, k in scored[:top_n]]

def select_top_judge_terms(keyword_counts: Dict[str, int], keyword_idf: Dict[str, float], top_n: int) -> List[str]:
    scored = []
    for kw, count in (keyword_counts or {}).items():
        nk = normalize_keyword(kw)
        if not nk or is_noise_keyword(nk):
            continue
        idf = keyword_idf.get(nk)
        if not idf:
            continue
        scored.append((count * idf, nk))
    scored.sort(reverse=True)
    return [k for _, k in scored[:top_n]]

def semantic_keyword_matches(
    case_keywords: List[str],
    judge_keyword_counts: Dict[str, int],
    keyword_idf: Dict[str, float],
    embed_model: SentenceTransformer
) -> List[Dict[str, Any]]:
    if not embed_model or not keyword_idf:
        return []
    case_terms = select_top_case_terms(case_keywords, keyword_idf, KEYWORD_TOP_CASE_TERMS)
    judge_terms = select_top_judge_terms(judge_keyword_counts, keyword_idf, KEYWORD_TOP_JUDGE_TERMS)
    if not case_terms or not judge_terms:
        return []

    case_embs = embed_model.encode(case_terms, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    judge_embs = embed_model.encode(judge_terms, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    sims = case_embs @ judge_embs.T

    pairs = []
    for i, ct in enumerate(case_terms):
        for j, jt in enumerate(judge_terms):
            pairs.append((float(sims[i, j]), ct, jt))
    pairs.sort(key=lambda x: x[0], reverse=True)

    out = []
    used_case = set()
    used_judge = set()
    for sim, ct, jt in pairs:
        if sim < KEYWORD_SIM_THRESHOLD:
            break
        if ct in used_case or jt in used_judge:
            continue
        out.append({"case_term": ct, "judge_term": jt, "similarity": sim})
        used_case.add(ct)
        used_judge.add(jt)
        if len(out) >= KEYWORD_SIM_TOP_N:
            break
    return out

def serialize_profiles(profiles: Dict[str, "JudgeProfile"]) -> Dict[str, Any]:
    out = {}
    for j, p in profiles.items():
        out[j] = {
            "embedding": p.embedding.tolist(),
            "n_train_cases": p.n_train_cases,
            "avg_turnaround_days": p.avg_turnaround_days,
            "median_turnaround_days": p.median_turnaround_days,
            "avg_complexity": p.avg_complexity,
            "area_counts": p.area_counts,
            "keyword_counts": p.keyword_counts,
        }
    return out

def debug_sample_case_features(cases: List[Dict[str, Any]], case_keywords: List[List[str]], limit: int = 10) -> None:
    """
    Print a small sample of judge/area/keywords to inspect noise.
    """
    print_block("DEBUG CASE FEATURES (sample)")
    n = min(limit, len(cases))
    for i in range(n):
        c = cases[i]
        kws = case_keywords[i] if i < len(case_keywords) else []
        print(f"- {c.get('case_id') or '[no id]'} | {c.get('title','')[:80]}")
        print(f"  URL: {c.get('case_url')}")
        print(f"  Judges: {', '.join(c.get('justices') or [])}")
        print(f"  Area: {c.get('area_of_law')}")
        print(f"  Keywords: {', '.join(kws[:12])}")
    print()

def deserialize_profiles(data: Dict[str, Any]) -> Dict[str, "JudgeProfile"]:
    out: Dict[str, "JudgeProfile"] = {}
    for j, p in (data or {}).items():
        out[j] = JudgeProfile(
            name=j,
            embedding=np.array(p.get("embedding", []), dtype=np.float32),
            n_train_cases=int(p.get("n_train_cases", 0)),
            avg_turnaround_days=p.get("avg_turnaround_days"),
            median_turnaround_days=p.get("median_turnaround_days"),
            avg_complexity=p.get("avg_complexity"),
            area_counts=p.get("area_counts") or {},
            keyword_counts=p.get("keyword_counts") or {},
        )
    return out

def load_derived_cache(cases: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not USE_DERIVED_CACHE or OVERWRITE_DERIVED_CACHE:
        return None
    if not os.path.exists(DERIVED_META_PATH) or not os.path.exists(DERIVED_ARRAYS_PATH):
        return None
    try:
        with open(DERIVED_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("version") != DERIVED_CACHE_VERSION:
            return None
        if meta.get("embed_model") != EMBED_MODEL_NAME:
            return None
        if meta.get("test_size") != TEST_SIZE:
            return None
        if meta.get("cases_hash") != compute_cases_hash(cases):
            return None
        arrays = np.load(DERIVED_ARRAYS_PATH)
        return {
            "case_embs": arrays["case_embs"],
            "case_complexity": arrays["case_complexity"],
            "case_keywords": meta.get("case_keywords") or [],
            "keyword_idf": meta.get("keyword_idf") or {},
            "train_idx": meta.get("train_idx") or [],
            "test_idx": meta.get("test_idx") or [],
            "profiles": deserialize_profiles(meta.get("profiles") or {}),
            "speed_scores": meta.get("speed_scores") or {},
            "train_by_judge": meta.get("train_by_judge") or {},
        }
    except Exception as exc:
        debug_log(f"[cache] Failed to load derived cache: {exc}")
        return None

def save_derived_cache(
    cases: List[Dict[str, Any]],
    case_embs: np.ndarray,
    case_complexity: np.ndarray,
    case_keywords: List[List[str]],
    keyword_idf: Dict[str, float],
    train_idx: List[int],
    test_idx: List[int],
    profiles: Dict[str, "JudgeProfile"],
    speed_scores: Dict[str, float],
    train_by_judge: Dict[str, List[int]]
) -> None:
    if not USE_DERIVED_CACHE:
        return
    meta = {
        "version": DERIVED_CACHE_VERSION,
        "cases_hash": compute_cases_hash(cases),
        "embed_model": EMBED_MODEL_NAME,
        "test_size": TEST_SIZE,
        "case_keywords": case_keywords,
        "keyword_idf": keyword_idf,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "profiles": serialize_profiles(profiles),
        "speed_scores": speed_scores,
        "train_by_judge": train_by_judge,
    }
    with open(DERIVED_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    case_embs = case_embs.astype(np.float32, copy=False)
    case_complexity = case_complexity.astype(np.float32, copy=False)
    np.savez_compressed(
        DERIVED_ARRAYS_PATH,
        case_embs=case_embs,
        case_complexity=case_complexity
    )


# ============================
# Crawling / Parsing
# ============================

def fetch_case_list_page(session: requests.Session, page: int) -> Tuple[List[str], Dict[str, Any]]:
    url = f"{BASE_CASES_LIST_URL}?p={page}"
    info = {
        "url": url,
        "page": page,
        "status": None,
        "html_len": 0,
        "html_sha1": None,
        "url_count": 0,
        "urls_hash": None,
        "init_pagination": None,
        "init_case_ids": [],
    }
    r = request_get(url, session)
    if r is None:
        info["status"] = None
        return [], info

    info["status"] = r.status_code
    html = r.text or ""
    info["html_len"] = len(html)
    info["html_sha1"] = sha1(html) if html else None
    info["init_pagination"] = extract_init_cases_pagination(html)
    info["init_case_ids"] = extract_init_case_ids(html)

    if r.status_code != 200:
        return [], info

    soup = BeautifulSoup(html, "lxml")
    urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # Typical case pages look like: /cases/uksc-2025-0043
        if href.startswith("/cases/uksc-") and "/judgments/" not in href:
            urls.add(BASE_CASE_URL + href)
    urls = sorted(urls)
    info["url_count"] = len(urls)
    info["urls_hash"] = sha1("\n".join(urls)) if urls else None
    return urls, info

def list_case_urls(session: requests.Session, page: int) -> List[str]:
    urls, _ = fetch_case_list_page(session, page)
    return urls

def list_case_urls_from_sitemap(session: requests.Session) -> List[str]:
    r = request_get(SITEMAP_URL, session)
    if r is None or r.status_code != 200:
        debug_log(f"[sitemap][warn] Failed to fetch sitemap: {SITEMAP_URL}")
        return []
    html = r.text or ""
    urls = re.findall(r'https?://(?:www\.)?supremecourt\.uk/cases/uksc-[^<\s]+', html)
    norm = []
    for u in urls:
        u = u.replace("https://supremecourt.uk", BASE_CASE_URL)
        u = u.replace("http://supremecourt.uk", BASE_CASE_URL)
        if "/judgments/" in u:
            continue
        norm.append(u)
    # unique, preserve order
    out = []
    seen = set()
    for u in norm:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    debug_log(f"[sitemap] extracted {len(out)} case URLs from {SITEMAP_URL}")
    if out:
        sample = ", ".join(out[:CRAWL_DEBUG_SAMPLE_IDS])
        debug_log(f"[sitemap] sample URLs: {sample}")
    return out

def list_highcourt_case_urls(session: requests.Session) -> List[str]:
    """
    BAILII listing: iterate toc-A.html ... toc-Z.html and collect case pages.
    """
    letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    urls: List[str] = []
    seen = set()
    for letter in letters:
        toc_url = HIGHCOURT_TOC_TEMPLATE.format(letter)
        r = request_get(toc_url, session, headers=BAILII_HEADERS)
        if r is None or r.status_code != 200:
            debug_log(f"[bailii][warn] Failed to fetch toc-{letter}: {toc_url}")
            continue
        html = r.text or ""
        soup = BeautifulSoup(html, "lxml")
        new_links = 0
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if "/ew/cases/EWHC/KB/" not in href:
                continue
            if not href.lower().endswith(".html"):
                continue
            full = href
            if href.startswith("/"):
                full = HIGHCOURT_BASE_URL.rstrip("/") + href
            if full in seen:
                continue
            seen.add(full)
            urls.append(full)
            new_links += 1
        debug_log(f"[bailii] toc-{letter}: +{new_links} (total {len(seen)})")
        time.sleep(REQUEST_DELAY_SEC)

    urls.sort()
    if urls and CRAWL_DEBUG:
        sample = ", ".join(urls[:CRAWL_DEBUG_SAMPLE_IDS])
        debug_log(f"[bailii] sample URLs: {sample}")
    return urls

def debug_crawl_page(info: Dict[str, Any], prev_info: Optional[Dict[str, Any]] = None) -> None:
    if not CRAWL_DEBUG:
        return
    sha = (info.get("html_sha1") or "")[:10]
    debug_log(
        f"[crawl] page={info.get('page')} url={info.get('url')} status={info.get('status')} "
        f"urls={info.get('url_count')} html_len={info.get('html_len')} html_sha1={sha}"
    )
    pag = info.get("init_pagination")
    if pag:
        debug_log(
            "[crawl] initCasesData pagination "
            f"page={pag.get('page')} page_count={pag.get('page_count')} "
            f"total={pag.get('total')} page_size={pag.get('page_size')}"
        )
        if pag.get("page") != info.get("page"):
            debug_log(
                f"[crawl][warn] initCasesData page={pag.get('page')} does not match requested page={info.get('page')}"
            )
    else:
        debug_log("[crawl][warn] initCasesData pagination not found in HTML")

    case_ids = info.get("init_case_ids") or []
    if case_ids:
        sample = ", ".join(case_ids[:CRAWL_DEBUG_SAMPLE_IDS])
        debug_log(f"[crawl] initCasesData caseId sample: {sample}")

    if prev_info:
        if info.get("html_sha1") and info.get("html_sha1") == prev_info.get("html_sha1"):
            debug_log("[crawl][warn] HTML identical to previous page; pagination may be ignored or JS-driven")
        if info.get("urls_hash") and info.get("urls_hash") == prev_info.get("urls_hash"):
            debug_log("[crawl][warn] Extracted case URLs identical to previous page")

def get_judgment_page_url(case_url: str) -> str:
    # case_url: https://www.supremecourt.uk/cases/uksc-2025-0043
    slug = case_url.rstrip("/").split("/")[-1]
    return f"{BASE_CASE_URL}/cases/judgments/{slug}"

def cache_path(case_url: str) -> str:
    return os.path.join(CASE_JSON_DIR, f"{sha1(case_url)}.json")

def extract_panel_judges(soup: BeautifulSoup) -> List[str]:
    """
    BAILII often wraps the sitting panel in <panel>...</panel>.
    Extract clean judge names from those tags.
    """
    judges = []
    for p in soup.find_all("panel"):
        txt = p.get_text("\n")
        parts = [t.strip() for t in txt.split("\n") if t.strip()]
        if not parts:
            continue
        # Typically first line is the judge name; drop role lines like "(Sitting as ...)"
        name = parts[0]
        name = re.sub(r"\s*\(.*?\)\s*$", "", name)
        name = re.sub(r"\s{2,}", " ", name).strip()
        if name:
            judges.append(name)
    return dedup_judges_ordered(judges)

def extract_bailii_judges(lines: List[str]) -> List[str]:
    """
    Heuristic extraction of judge names from BAILII page text.
    """
    judges = []
    stop_tokens = {
        "between", "and between", "claimant", "claimants", "defendant", "defendants",
        "applicant", "applicants", "respondent", "respondents", "appellant", "appellants",
        "hearing dates", "hearing date"
    }
    for i, ln in enumerate(lines[:300]):
        norm = re.sub(r"[^a-z]", "", ln.lower())
        if norm.startswith("before"):
            for j in range(i + 1, min(len(lines), i + 6)):
                cand = lines[j].strip(" :;\t")
                if not cand:
                    continue
                low = cand.lower()
                if any(tok in low for tok in stop_tokens):
                    break
                if len(cand) > 120:
                    continue
                if not re.search(r"[A-Za-z]", cand):
                    continue
                judges.append(cand)
            break

    if not judges:
        for ln in lines[:150]:
            low = ln.lower()
            if len(ln) > 80:
                continue
            if "adjudged" in low or "judgment" in low:
                continue
            if re.match(r"^(the hon\.?\s+)?(lord|lady|mr|mrs|ms|miss|sir|dame)\b.*(justice|judge)", ln, re.IGNORECASE):
                judges.append(ln.strip())
            elif re.search(r"\b(his|her)\s+honour\s+judge\b", low):
                judges.append(ln.strip())
            elif re.search(r"\bhhj\b", low) and "judge" in low:
                judges.append(ln.strip())
        judges = judges[:3]

    return dedup_judges_ordered(judges)

def parse_highcourt_case(case_url: str, session: requests.Session) -> Optional[Dict[str, Any]]:
    """
    Parse a BAILII High Court (KB) case page.
    """
    cpath = cache_path(case_url)
    if os.path.exists(cpath):
        try:
            with open(cpath, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("_parse_version") == CASE_PARSE_VERSION:
                return cached
        except Exception:
            pass

    r = request_get(case_url, session, headers=BAILII_HEADERS)
    if r is None or r.status_code != 200:
        return None

    soup = BeautifulSoup(r.text, "lxml")
    lines = extract_lines(soup)
    panel_judges = extract_panel_judges(soup)
    opinion_author = infer_opinion_author(lines)

    # Title
    title = clean_text(soup.title.get_text(" ")) if soup.title else None

    # Neutral citation / case id
    case_id = None
    citation_pattern = r"\[\d{4}\]\s*EWHC\s*\d+\s*\([A-Za-z]{2,5}\)"
    for ln in lines[:200]:
        m = re.search(citation_pattern, ln)
        if m:
            case_id = m.group(0)
            break
    if not case_id and title:
        m = re.search(citation_pattern, title)
        if m:
            case_id = m.group(0)
    if not case_id:
        # fallback using URL components
        parts = case_url.rstrip("/").split("/")
        if len(parts) >= 3:
            year = parts[-2] if parts[-2].isdigit() else None
            num = parts[-1].replace(".html", "")
            if year and num:
                case_id = f"[{year}] EWHC {num} (KB)"

    # Judgment date
    judgment_date = None
    for dt in soup.find_all("date"):
        jd = parse_date_maybe(dt.get_text(" "))
        if jd:
            judgment_date = jd
            break
    if not judgment_date and title:
        m = re.search(r"\((\d{1,2} [A-Za-z]+ \d{4})\)", title)
        if m:
            judgment_date = parse_date_maybe(m.group(1))
    if not judgment_date:
        for ln in lines[:120]:
            m = re.search(r"\b\d{1,2} [A-Za-z]+ \d{4}\b", ln)
            if m:
                judgment_date = parse_date_maybe(m.group(0))
                break

    justices = panel_judges
    if not justices:
        if PANEL_JUDGES_ONLY:
            debug_log(f"[bailii][warn] No <panel> judges found; skipping case {case_url}")
            return None
        justices = extract_bailii_judges(lines)
    if opinion_author and opinion_author not in justices:
        justices.append(opinion_author)
    # Deduplicate but preserve first entry order; keep only first name to avoid duplicates of same judge wording
    seen = set()
    dedup = []
    for j in justices:
        if j.lower() in seen:
            continue
        seen.add(j.lower())
        dedup.append(j)
    if dedup:
        justices = [dedup[0]]  # High Court cases are typically single-judge; keep the first parsed
    else:
        justices = []
    if CRAWL_DEBUG:
        debug_log(f"[bailii][judges] {case_url} -> {justices}")
        if any(len(j) > 80 for j in justices):
            debug_log(f"[bailii][warn] judge name too long; skipping {case_url}")
            return None

    text_full = soup_main_text(soup)
    analysis_text = text_full

    # Area of law: try to pick a division/court line near the top
    area_of_law = "High Court (King's Bench)"
    for ln in lines[:120]:
        if len(ln) > 80:
            continue
        if re.search(r"\bdivision\b", ln, flags=re.IGNORECASE) or re.search(r"\bcourt\b", ln, flags=re.IGNORECASE):
            area_of_law = ln.strip()
            break

    out = {
        "case_url": case_url,
        "judgment_url": case_url,
        "case_id": case_id or "",
        "title": title or "",
        "area_of_law": area_of_law,
        "issue": "",
        "facts": "",
        "date_of_issue": None,
        "judgment_date": judgment_date,
        "hearing_start": None,
        "hearing_end": None,
        "justices": justices,
        "opinion_author": opinion_author or "",
        "text_full": text_full[:2_000_000],
        "analysis_text": analysis_text[:2_000_000],
        "source_urls": {"case": case_url, "judgment": case_url},
        "_parse_version": CASE_PARSE_VERSION,
    }

    if not out.get("judgment_date"):
        return None

    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out

def parse_case(case_url: str, session: requests.Session) -> Optional[Dict[str, Any]]:
    """
    Parse both the /cases/<slug> page (metadata) and /cases/judgments/<slug> (full text if present).
    Returns dict with keys:
      case_id, title, area_of_law, issue, facts, date_of_issue, judgment_date,
      hearing_start, hearing_end, justices, text_full, source_urls
    """
    # Cache
    cpath = cache_path(case_url)
    if os.path.exists(cpath):
        try:
            with open(cpath, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("_parse_version") == CASE_PARSE_VERSION:
                return cached
        except Exception:
            pass

    # --- fetch case page ---
    r = request_get(case_url, session)
    if r is None or r.status_code != 200:
        return None
    soup_case = BeautifulSoup(r.text, "lxml")
    lines_case = extract_lines(soup_case)
    opinion_author = None

    # Title
    h1 = soup_case.find("h1")
    title = clean_text(h1.get_text(" ")) if h1 else None

    # Case ID
    # Often appears as UKSC/2025/0043 in text
    case_id = None
    for ln in lines_case[:200]:
        m = re.search(r"\bUKSC/\d{4}/\d{4}(?:/[A-Z])?\b", ln)
        if m:
            case_id = m.group(0)
            break
    # Area of law: often right after case id in early lines
    area_of_law = None
    if case_id:
        try:
            idx = lines_case.index(case_id)
            # find next uppercase-ish phrase within next few lines
            for j in range(idx + 1, min(idx + 10, len(lines_case))):
                cand = lines_case[j].strip()
                if cand and len(cand) < 80 and cand.upper() == cand and re.search(r"[A-Z]", cand):
                    area_of_law = cand
                    break
        except ValueError:
            pass

    issue = find_block_after_label(
        lines_case,
        "Issue",
        stop_labels=["Facts", "Date of issue", "Case origin", "Written arguments", "Statements of Facts and Issues", "Judgment details", "Appeal", "Change log"]
    )
    facts = find_block_after_label(
        lines_case,
        "Facts",
        stop_labels=["Date of issue", "Case origin", "Written arguments", "Statements of Facts and Issues", "Judgment details", "Appeal", "Change log"]
    )

    date_of_issue = parse_date_maybe(find_value_after_label(lines_case, "Date of issue") or "")
    judgment_date = parse_date_maybe(find_value_after_label(lines_case, "Judgment date") or "")

    hearing_start = parse_date_maybe(find_value_after_label(lines_case, "Start date") or "")
    hearing_end = parse_date_maybe(find_value_after_label(lines_case, "End date") or "")

    # --- judgment page (full text + reliable justices/hearing/judgment dates) ---
    judgment_url = get_judgment_page_url(case_url)
    justices = []
    text_full = ""
    judgment_date_j = None
    hearing_start_j = None
    hearing_end_j = None

    rj = request_get(judgment_url, session)
    if rj is not None and rj.status_code == 200:
        soup_j = BeautifulSoup(rj.text, "lxml")
        lines_j = extract_lines(soup_j)

        # override with judgment page metadata if present
        judgment_date_j = parse_date_maybe(find_value_after_label(lines_j, "Judgment date") or "")
        hearing_start_j = parse_date_maybe(find_value_after_label(lines_j, "Start date") or "")
        hearing_end_j = parse_date_maybe(find_value_after_label(lines_j, "End date") or "")
        opinion_author = infer_opinion_author(lines_j)

        # Justices: collect names after "Justices" until we hit "Judgment details"
        js = []
        for i in range(len(lines_j)):
            if lines_j[i].strip().lower() == "justices":
                for j in range(i + 1, min(i + 30, len(lines_j))):
                    if lines_j[j].strip().lower() in {"judgment details", "judgment date", "neutral citation", "hearing dates"}:
                        break
                    name = lines_j[j].strip()
                    if re.match(r"^(Lord|Lady)\s", name) or name in {"Lord Burnett"}:
                        js.append(name)
                break
        justices = sorted(set(js))

        # Full text from main
        text_full = soup_main_text(soup_j)

    # Decide final fields
    if judgment_date_j:
        judgment_date = judgment_date_j
    if hearing_start_j:
        hearing_start = hearing_start_j
    if hearing_end_j:
        hearing_end = hearing_end_j

    # If no justices found from judgment page, try case page "Justices" section (less reliable)
    if not justices:
        # collect a tags whose text looks like a justice name
        js = []
        for a in soup_case.find_all("a"):
            t = (a.get_text(" ") or "").strip()
            if re.match(r"^(Lord|Lady)\s", t) or t in {"Lord Burnett"}:
                js.append(t)
        justices = sorted(set(js))
    if not opinion_author:
        opinion_author = infer_opinion_author(lines_case)
    if opinion_author and opinion_author not in justices:
        justices.append(opinion_author)

    # Require a judgment date for this pipeline (since urgency profiling depends on it)
    if not judgment_date:
        return None

    # Build "analysis text" used for embedding and keywords
    # (use full judgment if available; else fall back to case summary sections)
    summary_text = " ".join([x for x in [title, area_of_law, issue, facts] if x])
    if text_full and len(text_full) > len(summary_text) + 500:
        analysis_text = text_full
    else:
        analysis_text = summary_text

    out = {
        "case_url": case_url,
        "judgment_url": judgment_url,
        "case_id": case_id or "",
        "title": title or "",
        "area_of_law": area_of_law or "",
        "issue": issue or "",
        "facts": facts or "",
        "date_of_issue": date_of_issue,
        "judgment_date": judgment_date,
        "hearing_start": hearing_start,
        "hearing_end": hearing_end,
        "justices": justices,
        "opinion_author": opinion_author or "",
        "text_full": text_full[:2_000_000],  # safety cap
        "analysis_text": analysis_text[:2_000_000],
        "source_urls": {"case": case_url, "judgment": judgment_url},
        "_parse_version": CASE_PARSE_VERSION,
    }

    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out

def crawl_cases(
    n_cases: int,
    session: requests.Session,
    cache_path: Optional[str] = None,
    seed_cases: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = list(seed_cases) if seed_cases else []
    seen: set = {c.get("case_url") for c in found if c.get("case_url")}
    if len(found) >= n_cases:
        return found[:n_cases]

    last_saved = len(found)
    parse_fn = parse_case if COURT_MODE == "supremecourt" else parse_highcourt_case

    def handle_urls(urls: List[str], progress_desc: Optional[str] = None) -> bool:
        nonlocal last_saved
        iterator = urls
        if progress_desc:
            iterator = tqdm(urls, desc=progress_desc)
        for u in iterator:
            if u in seen:
                continue
            seen.add(u)

            case = parse_fn(u, session)
            time.sleep(REQUEST_DELAY_SEC)

            if case is None:
                continue

            # Ensure we have at least: id/title/judgment date/justices
            if not case.get("case_id") or not case.get("title") or not case.get("justices"):
                # still keep (some older entries might be sparse), but warn
                pass

            found.append(case)

            if cache_path and (len(found) - last_saved) >= CACHE_SAVE_EVERY:
                save_cases_cache(cache_path, found)
                last_saved = len(found)

            if len(found) >= n_cases:
                if cache_path:
                    save_cases_cache(cache_path, found)
                return True
        return False

    if COURT_MODE == "highcourt":
        print_block("HIGH COURT LISTING")
        urls = list_highcourt_case_urls(session)
        if urls:
            if handle_urls(urls, progress_desc="Parsing High Court cases"):
                return found
        else:
            debug_log("[bailii][warn] No High Court case URLs extracted; returning cached/seed cases.")
        if cache_path:
            save_cases_cache(cache_path, found)
        return found

    use_sitemap = CASE_LISTING_MODE == "sitemap"
    auto_fallback = CASE_LISTING_MODE == "auto"

    if not use_sitemap:
        prev_info: Optional[Dict[str, Any]] = None
        print_block("LISTING PAGES")
        for page in tqdm(range(1, MAX_LIST_PAGES + 1), desc="Listing pages"):
            urls, info = fetch_case_list_page(session, page)
            debug_crawl_page(info, prev_info)

            if auto_fallback and page > 1:
                pag = info.get("init_pagination") or {}
                if (pag and pag.get("page") != page) or (
                    prev_info and info.get("html_sha1") == prev_info.get("html_sha1")
                ):
                    debug_log("[crawl][warn] Pagination appears static; switching to sitemap crawl.")
                    use_sitemap = True
                    break

            if not urls:
                debug_log(f"[crawl][warn] No case URLs extracted for page {page}")
                prev_info = info
                continue

            if handle_urls(urls):
                return found

            prev_info = info

    if use_sitemap:
        print_block("SITEMAP LISTING")
        urls = list_case_urls_from_sitemap(session)
        if urls:
            if handle_urls(urls, progress_desc="Parsing sitemap cases"):
                return found
        else:
            debug_log("[sitemap][warn] No case URLs extracted; returning what we have.")

    if cache_path:
        save_cases_cache(cache_path, found)
    return found


# ============================
# Embedding + Complexity + Keywords
# ============================

def build_embeddings(cases: List[Dict[str, Any]], model: SentenceTransformer) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Returns:
      case_embeddings: (N, D)
      case_to_chunk_idxs: mapping case_idx -> list of chunk indices
    """
    all_chunks = []
    case_to_chunk_idxs = {}
    for i, c in enumerate(cases):
        chunks = chunk_words(c.get("analysis_text", ""), CHUNK_WORDS, CHUNK_OVERLAP)
        if not chunks:
            chunks = [c.get("analysis_text", "")[:2000] or c.get("title", "")]
        start = len(all_chunks)
        all_chunks.extend(chunks)
        case_to_chunk_idxs[i] = list(range(start, start + len(chunks)))

    # Encode
    chunk_embs = model.encode(
        all_chunks,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Pool to case embedding
    case_embs = []
    for i in range(len(cases)):
        idxs = case_to_chunk_idxs[i]
        ce = chunk_embs[idxs].mean(axis=0)
        ce = ce / (np.linalg.norm(ce) + 1e-12)
        case_embs.append(ce)

    return np.vstack(case_embs), case_to_chunk_idxs

def compute_complexity(cases: List[Dict[str, Any]], case_embs: np.ndarray, model: SentenceTransformer) -> np.ndarray:
    """
    A lightweight complexity proxy:
      - log(length in words)
      - topical dispersion: average cosine distance of chunks from doc centroid
    Produces normalized [0,1] score.
    """
    # Re-derive chunk embeddings cheaply by re-chunking + encoding smaller sample:
    # We'll approximate dispersion using 6 representative chunks per doc (head/mid/tail).
    disp = []
    lengths = []
    for c in tqdm(cases, desc="Complexity features"):
        text = c.get("analysis_text", "")
        words = text.split()
        lengths.append(len(words))
        chunks = chunk_words(text, CHUNK_WORDS, CHUNK_OVERLAP)
        if len(chunks) <= 1:
            disp.append(0.0)
            continue
        # sample up to 6 chunks
        pick = []
        for frac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            idx = min(len(chunks)-1, int(frac * (len(chunks)-1)))
            pick.append(chunks[idx])
        pick = list(dict.fromkeys(pick))  # unique
        em = model.encode(pick, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
        centroid = em.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        # average cosine distance
        d = float(np.mean([1.0 - cosine_sim(centroid, e) for e in em]))
        disp.append(d)

    lengths = np.array(lengths, dtype=np.float32)
    disp = np.array(disp, dtype=np.float32)

    loglen = np.log1p(lengths)

    # z-score then squash to [0,1]
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    raw = 0.65 * z(loglen) + 0.35 * z(disp)
    # logistic to 0..1
    comp = 1.0 / (1.0 + np.exp(-raw))
    return comp.astype(np.float32)

def extract_case_keywords(cases: List[Dict[str, Any]], kw_model: KeyBERT, top_n: int = 10) -> List[List[str]]:
    """
    Extract keywords from the full analysis_text (fallback to issue/facts).
    """
    out = []
    for c in tqdm(cases, desc="Keywords"):
        seed = (c.get("analysis_text","") or "").strip()
        if not seed:
            seed = " ".join([c.get("area_of_law",""), c.get("issue",""), c.get("facts","")]).strip()
        seed = seed[:12000]  # cap for speed
        try:
            kws = kw_model.extract_keywords(
                seed,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                top_n=top_n,
                use_mmr=True,
                diversity=0.5
            )
            filtered = [k for k, _ in kws if not is_noise_keyword(k)]
            out.append(filtered)
        except Exception:
            out.append([])
    return out


# ============================
# Judge Profiling + Assignment
# ============================

@dataclass
class JudgeProfile:
    name: str
    embedding: np.ndarray
    n_train_cases: int
    avg_turnaround_days: Optional[float]
    median_turnaround_days: Optional[float]
    avg_complexity: Optional[float]
    area_counts: Dict[str, int]
    keyword_counts: Dict[str, int]

def build_judge_profiles(
    cases: List[Dict[str, Any]],
    case_embs: np.ndarray,
    case_complexity: np.ndarray,
    case_keywords: List[List[str]],
    train_indices: List[int]
) -> Dict[str, JudgeProfile]:
    by_judge_emb_sum = defaultdict(lambda: np.zeros(case_embs.shape[1], dtype=np.float32))
    by_judge_emb_w = defaultdict(float)
    by_judge_turn = defaultdict(list)   # list of (days, weight)
    by_judge_comp = defaultdict(list)   # list of (comp, weight)
    by_judge_area = defaultdict(Counter)  # weighted counts
    by_judge_kw = defaultdict(Counter)    # weighted counts

    for idx in train_indices:
        c = cases[idx]
        emb = case_embs[idx]
        comp = float(case_complexity[idx])
        tdays = days_between(c.get("hearing_start"), c.get("judgment_date"))
        area = (c.get("area_of_law") or "").strip()
        kws = case_keywords[idx] or []
        op_author = normalize_person_name(c.get("opinion_author") or "")

        for j in c.get("justices", []) or []:
            w = 1.0
            if USE_OPINION_AUTHOR_WEIGHTING and op_author and normalize_person_name(j) == op_author:
                w = OPINION_AUTHOR_WEIGHT

            by_judge_emb_sum[j] += emb * w
            by_judge_emb_w[j] += w
            by_judge_comp[j].append((comp, w))
            if tdays is not None and tdays >= 0:
                by_judge_turn[j].append((tdays, w))
            if area:
                by_judge_area[j][area] += w
            for kw in kws:
                nk = normalize_keyword(kw)
                if nk:
                    by_judge_kw[j][nk] += w

    profiles = {}
    for j, w_sum in by_judge_emb_w.items():
        if w_sum <= 0:
            continue
        centroid = by_judge_emb_sum[j] / w_sum
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        turns = by_judge_turn.get(j, [])
        avg_t = None
        med_t = None
        if turns:
            vals = [t for t, _ in turns]
            weights = [w for _, w in turns]
            avg_t = float(np.average(vals, weights=weights))
            med_t = float(np.median(vals))

        avg_c = None
        if by_judge_comp[j]:
            comp_vals = [v for v, _ in by_judge_comp[j]]
            comp_w = [w for _, w in by_judge_comp[j]]
            avg_c = float(np.average(comp_vals, weights=comp_w))

        profiles[j] = JudgeProfile(
            name=j,
            embedding=centroid,
            n_train_cases=int(round(w_sum)),
            avg_turnaround_days=avg_t,
            median_turnaround_days=med_t,
            avg_complexity=avg_c,
            area_counts=dict(by_judge_area[j]),
            keyword_counts=dict(by_judge_kw[j]),
        )
    return profiles

def normalize_speed_scores(profiles: Dict[str, JudgeProfile]) -> Dict[str, float]:
    """
    Convert avg_turnaround_days to a normalized speed score (higher = faster).
    Uses 1/(days+1) then z-score then logistic.
    """
    vals = []
    keys = []
    for j, p in profiles.items():
        if p.avg_turnaround_days is not None and p.avg_turnaround_days >= 0:
            keys.append(j)
            vals.append(1.0 / (p.avg_turnaround_days + 1.0))
    if not vals:
        return {j: 0.0 for j in profiles.keys()}

    v = np.array(vals, dtype=np.float32)
    z = (v - v.mean()) / (v.std() + 1e-6)
    s = 1.0 / (1.0 + np.exp(-z))
    out = {j: 0.0 for j in profiles.keys()}
    for j, sc in zip(keys, s.tolist()):
        out[j] = float(sc)
    return out

def score_judges_for_case(
    case_idx: int,
    cases: List[Dict[str, Any]],
    case_embs: np.ndarray,
    case_complexity: np.ndarray,
    profiles: Dict[str, JudgeProfile],
    speed_scores: Dict[str, float]
) -> List[Dict[str, Any]]:
    emb = case_embs[case_idx]
    comp = float(case_complexity[case_idx])
    urgency = estimate_case_urgency(" ".join([cases[case_idx].get("issue", ""), cases[case_idx].get("facts", "")]))

    scores = []
    for j, p in profiles.items():
        sim = cosine_sim(emb, p.embedding)
        spd = speed_scores.get(j, 0.0)

        # complexity fit: prefer judges whose historical avg complexity is close
        if p.avg_complexity is None:
            cfit = 0.5
        else:
            cfit = 1.0 - min(1.0, abs(comp - float(p.avg_complexity)))

        w_speed = W_SPEED + URGENT_SPEED_BOOST * urgency
        score = (W_SIM * sim) + (w_speed * spd) + (W_COMPLEXITY_FIT * cfit)
        scores.append({
            "judge": j,
            "score": float(score),
            "sim": float(sim),
            "spd": float(spd),
            "cfit": float(cfit),
            "urgency": float(urgency),
            "w_speed": float(w_speed),
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores

def assign_judge(
    case_idx: int,
    cases: List[Dict[str, Any]],
    case_embs: np.ndarray,
    case_complexity: np.ndarray,
    profiles: Dict[str, JudgeProfile],
    speed_scores: Dict[str, float],
    train_indices_by_judge: Dict[str, List[int]],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Returns top_k (judge, score) sorted desc.
    """
    scores = score_judges_for_case(case_idx, cases, case_embs, case_complexity, profiles, speed_scores)
    return [(s["judge"], s["score"]) for s in scores[:top_k]]

def evaluate_assignments(
    test_indices: List[int],
    cases: List[Dict[str, Any]],
    case_embs: np.ndarray,
    case_complexity: np.ndarray,
    profiles: Dict[str, JudgeProfile],
    speed_scores: Dict[str, float],
    train_indices_by_judge: Dict[str, List[int]],
    top_k: int = 3
) -> Dict[str, float]:
    """
    Top-1 / Top-3 accuracy against the true panel membership.
    """
    top1 = 0
    topk = 0
    mrr = 0.0
    overlap_cases = 0
    overlap_total = 0
    n = 0

    for idx in test_indices:
        true_panel = set(cases[idx].get("justices", []) or [])
        if not true_panel:
            continue
        preds = assign_judge(idx, cases, case_embs, case_complexity, profiles, speed_scores, train_indices_by_judge, top_k=top_k)
        ranked = [j for j, _ in preds]

        n += 1
        if ranked and ranked[0] in true_panel:
            top1 += 1
        overlap = len(set(ranked[:top_k]).intersection(true_panel))
        if overlap > 0:
            topk += 1
            overlap_cases += 1
        overlap_total += overlap

        # MRR for membership
        rr = 0.0
        for r, j in enumerate(ranked, start=1):
            if j in true_panel:
                rr = 1.0 / r
                break
        mrr += rr

    if n == 0:
        return {"n": 0, "top1": 0.0, f"top{top_k}": 0.0, "mrr": 0.0, "overlap_cases": 0, "avg_overlap": 0.0}
    return {
        "n": n,
        "top1": top1 / n,
        f"top{top_k}": topk / n,
        "mrr": mrr / n,
        "overlap_cases": overlap_cases,
        "avg_overlap": overlap_total / n
    }

def build_test_report(
    assignments: List[Dict[str, Any]],
    cases: List[Dict[str, Any]],
    top_k: int = 3
) -> Dict[str, Any]:
    n_total = len(assignments)
    top1_count = 0
    topk_count = 0
    mrr_sum = 0.0
    overlap_cases = 0
    overlap_total = 0
    panel_sizes = []
    overlap_ratios = []
    overlap_dist = Counter()
    top1_pred_counts = Counter()
    top1_correct_counts = Counter()
    per_case = []

    for a in assignments:
        case_idx = a.get("case_idx")
        c = cases[case_idx]
        true_panel = a.get("true_panel") or []
        true_set = set(true_panel)
        preds = a.get("predictions") or []
        ranked = [p.get("judge") for p in preds if p.get("judge")]

        top1_pred = ranked[0] if ranked else None
        if top1_pred:
            top1_pred_counts[top1_pred] += 1

        panel_size = len(true_set)
        has_panel = panel_size > 0
        overlap = len(set(ranked[:top_k]).intersection(true_set)) if has_panel else 0
        overlap_dist[overlap] += 1

        rank_first = None
        if has_panel:
            panel_sizes.append(panel_size)
            for i, j in enumerate(ranked, start=1):
                if j in true_set:
                    rank_first = i
                    break
            if rank_first:
                mrr_sum += 1.0 / rank_first
            if ranked and ranked[0] in true_set:
                top1_count += 1
                top1_correct_counts[ranked[0]] += 1
            if overlap > 0:
                topk_count += 1
                overlap_cases += 1
            overlap_total += overlap
            overlap_ratio = overlap / panel_size if panel_size else 0.0
            overlap_ratios.append(overlap_ratio)
        else:
            overlap_ratio = None

        per_case.append({
            "case_idx": case_idx,
            "case_id": c.get("case_id"),
            "title": c.get("title"),
            "case_url": c.get("case_url"),
            "judgment_date": c.get("judgment_date"),
            "true_panel": true_panel,
            "true_panel_size": panel_size,
            "predictions": preds,
            "top1_pred": top1_pred,
            "top1_correct": bool(has_panel and ranked and ranked[0] in true_set),
            f"top{top_k}_hit": bool(has_panel and overlap > 0),
            "overlap_count": overlap if has_panel else None,
            "overlap_ratio": overlap_ratio,
            "rank_first_correct": rank_first
        })

    n_with_panel = len(panel_sizes)
    if n_with_panel:
        top1_acc = top1_count / n_with_panel
        topk_acc = topk_count / n_with_panel
        mrr = mrr_sum / n_with_panel
        avg_overlap = overlap_total / n_with_panel
        topk_precision = overlap_total / (n_with_panel * top_k)
        avg_panel_size = float(np.mean(panel_sizes))
        median_panel_size = float(np.median(panel_sizes))
        avg_overlap_ratio = float(np.mean(overlap_ratios)) if overlap_ratios else 0.0
    else:
        top1_acc = topk_acc = mrr = avg_overlap = topk_precision = 0.0
        avg_panel_size = median_panel_size = avg_overlap_ratio = 0.0

    metrics = {
        "n_total": n_total,
        "n_with_panel": n_with_panel,
        "top1_accuracy": top1_acc,
        f"top{top_k}_accuracy": topk_acc,
        "mrr": mrr,
        "overlap_cases": overlap_cases,
        "avg_overlap_count": avg_overlap,
        f"top{top_k}_precision": topk_precision,
        "avg_panel_size": avg_panel_size,
        "median_panel_size": median_panel_size,
        "avg_overlap_ratio": avg_overlap_ratio,
        "overlap_distribution": {str(k): v for k, v in sorted(overlap_dist.items())},
        "top1_prediction_counts": top1_pred_counts.most_common(),
        "top1_correct_counts": top1_correct_counts.most_common()
    }

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "max_cases": MAX_CASES,
            "test_size": TEST_SIZE,
            "embed_model": EMBED_MODEL_NAME,
            "weights": {"sim": W_SIM, "speed": W_SPEED, "complexity_fit": W_COMPLEXITY_FIT},
            "urgent_speed_boost": URGENT_SPEED_BOOST,
            "top_k": top_k
        },
        "metrics": metrics,
        "per_case": per_case
    }
    return report

def build_train_indices_by_judge(cases: List[Dict[str, Any]], train_indices: List[int]) -> Dict[str, List[int]]:
    m = defaultdict(list)
    for idx in train_indices:
        for j in cases[idx].get("justices", []) or []:
            m[j].append(idx)
    return m


# ============================
# Explanation Module
# ============================

def explain_assignment(
    case_idx: int,
    chosen_judge: str,
    cases: List[Dict[str, Any]],
    case_embs: np.ndarray,
    case_complexity: np.ndarray,
    case_keywords: List[List[str]],
    profiles: Dict[str, JudgeProfile],
    speed_scores: Dict[str, float],
    train_indices_by_judge: Dict[str, List[int]],
    keyword_idf: Optional[Dict[str, float]] = None,
    embed_model: Optional[SentenceTransformer] = None,
    top_assignments: int = 5,
    top_similar: int = 3
) -> Dict[str, Any]:
    c = cases[case_idx]
    prof = profiles[chosen_judge]

    emb = case_embs[case_idx]
    sim = cosine_sim(emb, prof.embedding)
    spd = speed_scores.get(chosen_judge, 0.0)

    comp = float(case_complexity[case_idx])
    cfit = 0.5 if prof.avg_complexity is None else 1.0 - min(1.0, abs(comp - float(prof.avg_complexity)))

    urgency = estimate_case_urgency(" ".join([c.get("issue",""), c.get("facts","")]))

    # Judge expertise areas
    areas_sorted = sorted(prof.area_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # Similar past cases for this judge (training)
    sims = []
    for tidx in train_indices_by_judge.get(chosen_judge, []):
        s = cosine_sim(emb, case_embs[tidx])
        sims.append((tidx, s))
    sims.sort(key=lambda x: x[1], reverse=True)
    top_cases = []
    for tidx, s in sims[:top_similar]:
        tc = cases[tidx]
        top_cases.append({
            "case_id": tc.get("case_id"),
            "title": tc.get("title"),
            "area_of_law": tc.get("area_of_law"),
            "judgment_date": tc.get("judgment_date"),
            "case_url": tc.get("case_url"),
            "similarity": float(s),
        })

    semantic_matches = []
    if embed_model and keyword_idf:
        semantic_matches = semantic_keyword_matches(
            case_keywords[case_idx] or [],
            prof.keyword_counts,
            keyword_idf,
            embed_model
        )

    reasons = [
        {"factor": "expertise_similarity", "value": sim, "detail": "Cosine similarity between case embedding and judge profile embedding."},
        {"factor": "speed_fit", "value": spd, "detail": "Normalized speed score from historical hearing→judgment turnaround (higher = faster)."},
        {"factor": "complexity_fit", "value": cfit, "detail": "How close the case complexity is to this judge’s historical average complexity."},
        {"factor": "urgency_estimate", "value": urgency, "detail": "Heuristic urgency score from issue/facts (used to upweight speed)."},
    ]

    explanation = {
        "case": {
            "case_id": c.get("case_id"),
            "title": c.get("title"),
            "area_of_law": c.get("area_of_law"),
            "judgment_date": c.get("judgment_date"),
            "hearing_start": c.get("hearing_start"),
            "case_url": c.get("case_url"),
            "true_panel": c.get("justices"),
        },
        "assigned_judge": {
            "name": chosen_judge,
            "n_train_cases": prof.n_train_cases,
            "avg_turnaround_days": prof.avg_turnaround_days,
            "median_turnaround_days": prof.median_turnaround_days,
            "avg_case_complexity": prof.avg_complexity,
            "top_areas_of_law": areas_sorted,
        },
        "why": reasons,
        "supporting_evidence": {
            "semantic_keyword_matches": semantic_matches,
            "most_similar_past_cases_for_judge": top_cases
        },
        "top_assignments": score_judges_for_case(
            case_idx,
            cases,
            case_embs,
            case_complexity,
            profiles,
            speed_scores
        )[:top_assignments]
    }
    return explanation

def print_explanation(expl: Dict[str, Any]) -> None:
    c = expl["case"]
    j = expl["assigned_judge"]
    print("\n" + "="*90)
    print(f"CASE: {c['case_id']} — {c['title']}")
    print(f"Area: {c.get('area_of_law','')}")
    print(f"Hearing start: {c.get('hearing_start')} | Judgment: {c.get('judgment_date')}")
    print(f"URL: {c.get('case_url')}")
    print(f"True panel: {', '.join(c.get('true_panel') or [])}")
    print("-"*90)
    print(f"ASSIGNED JUDGE: {j['name']}")
    print(f"Train cases: {j['n_train_cases']} | Avg turnaround days: {j['avg_turnaround_days']} | Median: {j['median_turnaround_days']}")
    print(f"Avg complexity handled: {j['avg_case_complexity']}")
    if j.get("top_areas_of_law"):
        areas = ", ".join([f"{a}({n})" for a, n in j["top_areas_of_law"][:3]])
        print(f"Top areas: {areas}")
    print("-"*90)
    for r in expl["why"]:
        print(f"{r['factor']}: {r['value']:.4f} — {r['detail']}")
    print("-"*90)
    top = expl.get("top_assignments") or []
    if top:
        print("Top assignments (score components):")
        header = f"{'rk':>2}  {'judge':<30} {'score':>7} {'sim':>6} {'spd':>6} {'cfit':>6} {'wspd':>6}"
        print(header)
        print("-"*90)
        for i, row in enumerate(top, start=1):
            name = row.get("judge", "")
            if len(name) > 30:
                name = f"{name[:28]}.."
            print(
                f"{i:>2}  {name:<30} {row['score']:>7.4f} {row['sim']:>6.3f} {row['spd']:>6.3f} "
                f"{row['cfit']:>6.3f} {row['w_speed']:>6.3f}"
            )
        print("-"*90)
    ov = expl["supporting_evidence"].get("semantic_keyword_matches") or []
    if ov:
        print("Semantic keyword matches:")
        for item in ov:
            print(f"  - {item['case_term']} ~ {item['judge_term']} (sim={item['similarity']:.2f})")
    sims = expl["supporting_evidence"]["most_similar_past_cases_for_judge"]
    if sims:
        print("\nMost similar past cases for this judge:")
        for s in sims:
            print(f"  - {s['case_id']} | sim={s['similarity']:.3f} | {s['title']}  ({s.get('area_of_law','')})")
            print(f"    {s['case_url']}")
    print("="*90 + "\n")


# ============================
# Main run
# ============================

def main():
    session = requests.Session()
    model = None

    print_block("CASE INGESTION")
    print(f"court_mode={COURT_MODE} | source={CASE_SOURCE_NAME}")
    print(f"cache_dir={CACHE_DIR}")
    seed_cases = []
    if USE_CACHED_CASES:
        seed_cases = load_cases_cache(CASES_CACHE_PATH)
        if seed_cases:
            print(f"Loaded {len(seed_cases)} cached cases from: {CASES_CACHE_PATH}")

    if len(seed_cases) >= MAX_CASES:
        cases = seed_cases[:MAX_CASES]
        print("Using cached cases; skipping crawl.")
    else:
        if seed_cases:
            print("Crawling cases (with judgment dates), resuming from cache...")
        else:
            print("Crawling cases (with judgment dates)...")
        cases = crawl_cases(MAX_CASES, session, cache_path=CASES_CACHE_PATH, seed_cases=seed_cases)
        print(f"Collected {len(cases)} cases.")

    if len(cases) < 30:
        print("Too few cases collected. Consider increasing MAX_LIST_PAGES, disabling USE_CACHED_CASES, or checking connectivity.")
        return

    # Save raw dataset
    print_block("SAVING RAW CASES")
    save_cases_cache(CASES_CACHE_PATH, cases)
    print(f"Saved raw cases to: {CASES_CACHE_PATH}")
    sync_export_files()

    print_block("FEATURE EXTRACTION / CACHES")
    derived = load_derived_cache(cases)
    if derived:
        if derived["case_embs"].shape[0] != len(cases) or derived["case_complexity"].shape[0] != len(cases):
            debug_log("[cache] Derived cache case count mismatch; recomputing.")
            derived = None
    if derived and len(derived.get("case_keywords") or []) != len(cases):
        debug_log("[cache] Derived cache keyword count mismatch; recomputing.")
        derived = None
    if derived:
        print(f"Loaded derived cache from: {DERIVED_META_PATH}")
        case_embs = derived["case_embs"]
        case_complexity = derived["case_complexity"]
        case_keywords = derived["case_keywords"]
        keyword_idf = derived["keyword_idf"]
        train_idx = derived["train_idx"]
        test_idx = derived["test_idx"]
        profiles = derived["profiles"]
        speed_scores = derived["speed_scores"]
        train_by_judge = derived.get("train_by_judge") or build_train_indices_by_judge(cases, train_idx)
        if not keyword_idf:
            keyword_idf = compute_keyword_idf(case_keywords)
        print(f"Train cases: {len(train_idx)} | Test cases: {len(test_idx)}")
        print(f"Loaded profiles for {len(profiles)} judges.")
        if CRAWL_DEBUG:
            debug_sample_case_features(cases, case_keywords, limit=10)
    else:
        # Embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model on {device}: {EMBED_MODEL_NAME}")
        model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

        print_block("EMBEDDINGS")
        print("Building embeddings...")
        case_embs, _ = build_embeddings(cases, model)
        print(f"Embeddings shape: {case_embs.shape}")

        print_block("COMPLEXITY")
        print("Computing complexity...")
        case_complexity = compute_complexity(cases, case_embs, model)

        # Keywords
        print_block("KEYWORDS")
        print("Extracting keywords (KeyBERT)...")
        kw_model = KeyBERT(model=model)
        case_keywords = extract_case_keywords(cases, kw_model, top_n=10)
        keyword_idf = compute_keyword_idf(case_keywords)
        if CRAWL_DEBUG:
            debug_sample_case_features(cases, case_keywords, limit=10)

        # Train/test split
        print_block("SPLIT + PROFILES")
        idxs = list(range(len(cases)))
        train_idx, test_idx = train_test_split(idxs, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        train_idx = sorted(train_idx)
        test_idx = sorted(test_idx)
        print(f"Train cases: {len(train_idx)} | Test cases: {len(test_idx)}")

        # Judge profiles
        profiles = build_judge_profiles(cases, case_embs, case_complexity, case_keywords, train_idx)
        print(f"Built profiles for {len(profiles)} judges.")

        # Speed score normalization
        speed_scores = normalize_speed_scores(profiles)

        # Judge -> training indices mapping
        train_by_judge = build_train_indices_by_judge(cases, train_idx)

        save_derived_cache(
            cases,
            case_embs,
            case_complexity,
            case_keywords,
            keyword_idf,
            train_idx,
            test_idx,
            profiles,
            speed_scores,
            train_by_judge
        )

    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model on {device}: {EMBED_MODEL_NAME}")
        model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    print_block("ASSIGNMENTS")
    assignments = []
    for idx in tqdm(test_idx, desc="Assigning test cases"):
        preds = assign_judge(idx, cases, case_embs, case_complexity, profiles, speed_scores, train_by_judge, top_k=3)
        assignments.append({
            "case_idx": idx,
            "case_id": cases[idx].get("case_id"),
            "title": cases[idx].get("title"),
            "true_panel": cases[idx].get("justices"),
            "predictions": [{"judge": j, "score": s} for j, s in preds]
        })

    out_path = os.path.join(EXPORT_DIR, "test_assignments.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(assignments, f, ensure_ascii=False, indent=2)
    print(f"\nSaved test assignments to: {out_path}")

    print_block("EVALUATION")
    report = build_test_report(assignments, cases, top_k=3)
    print("\n=== Evaluation (predict judge ∈ true panel) ===")
    print(json.dumps(report["metrics"], indent=2))
    if report["metrics"].get("n_with_panel", 0):
        overlap_cases = report["metrics"].get("overlap_cases", 0)
        overlap_rate = overlap_cases / report["metrics"]["n_with_panel"]
        avg_overlap = report["metrics"].get("avg_overlap_count", 0.0)
        print(
            f"Overlap summary: {overlap_cases}/{report['metrics']['n_with_panel']} cases overlap with true panel "
            f"({overlap_rate:.1%}); avg overlap per case={avg_overlap:.2f}"
        )

    with open(TEST_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved test report to: {TEST_REPORT_PATH}")

    # Save judge profiles (serializable)
    print_block("SAVING PROFILES")
    prof_out = {}
    for j, p in profiles.items():
        prof_out[j] = {
            "n_train_cases": p.n_train_cases,
            "avg_turnaround_days": p.avg_turnaround_days,
            "median_turnaround_days": p.median_turnaround_days,
            "avg_complexity": p.avg_complexity,
            "top_areas_of_law": sorted(p.area_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "top_keywords": Counter(p.keyword_counts).most_common(25),
        }
    prof_path = os.path.join(EXPORT_DIR, "judge_profiles_summary.json")
    with open(prof_path, "w", encoding="utf-8") as f:
        json.dump(prof_out, f, ensure_ascii=False, indent=2)
    print(f"Saved judge profile summaries to: {prof_path}")
    sync_export_files()

    # Zip outputs for easy sharing
    zip_path = zip_outputs(CACHE_DIR, ZIP_RESULTS)
    if zip_path:
        print_block("ZIP EXPORT")
        print(f"Zipped outputs to: {zip_path}")

    # Show a few explanations
    print_block("EXPLANATIONS")
    print("\nShowing example explanations for 3 random test cases...\n")
    for idx in random.sample(test_idx, k=min(3, len(test_idx))):
        preds = assign_judge(idx, cases, case_embs, case_complexity, profiles, speed_scores, train_by_judge, top_k=1)
        chosen = preds[0][0]
        expl = explain_assignment(
            idx,
            chosen,
            cases,
            case_embs,
            case_complexity,
            case_keywords,
            profiles,
            speed_scores,
            train_by_judge,
            keyword_idf=keyword_idf,
            embed_model=model,
            top_similar=3
        )
        print_explanation(expl)

    print("\nDone.")
    print(f"Cache directory: {CACHE_DIR}")

main()
