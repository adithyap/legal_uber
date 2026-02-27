#!/usr/bin/env python3
"""
EWHC (KB) turnaround-speed analysis
===================================
Goal: test whether judges differ systematically in hearing->judgment speed.
Outputs:
  - data/cases_speed.json      : cleaned cases with turnaround_days and controls
  - data/judge_effects.json    : judge fixed-effect estimates + descriptive stats
  - data/speed_results.json    : model fit, F-test for judge effects, dispersion metrics

This replaces the prior judge-identity prediction pipeline with a simpler, classical
fixed-effects setup: speed_ij = alpha + gamma_judge + delta_year + epsilon_ij.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from dateutil import parser as dateparser
import matplotlib.pyplot as plt

try:
    from scipy.stats import f as f_dist
except Exception:  # scipy may not be available in lightweight envs
    f_dist = None

# -----------------------------
# Config
# -----------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

YEARS_BEFORE = 25
CURRENT_YEAR = datetime.utcnow().year
MIN_CASE_YEAR = CURRENT_YEAR - YEARS_BEFORE

MAX_CASES = 2000
REQUEST_DELAY_SEC = 0.2
HTTP_TIMEOUT = 30

CASE_PARSE_VERSION = 7  # bump when parse logic changes
CACHE_DIR = Path("data/cache_highcourt")
CASE_JSON_DIR = CACHE_DIR / "case_json"
CASES_CACHE_PATH = CACHE_DIR / "cases_raw.json"
CACHE_SAVE_EVERY = 20

OUTPUT_DIR = Path("data")
CASES_OUTPUT_PATH = OUTPUT_DIR / "cases_speed.json"
JUDGE_EFFECTS_PATH = OUTPUT_DIR / "judge_effects.json"
SPEED_RESULTS_PATH = OUTPUT_DIR / "speed_results.json"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Analysis options
WINSOR_HIGH_PCTL = 0.99          # clip turnaround at this upper percentile (set to None to disable)
WINSOR_LOW_PCTL = 0.0            # lower percentile clip (keep 0 for turnaround)
USE_LOG_MODEL = True             # also run model on log(turnaround_days + 1)
INCLUDE_HEARING_YEAR = True      # add hearing_year fixed effects
INCLUDE_JUDGMENT_YEAR = True     # keep judgment_year fixed effects
INCLUDE_AREA_FE = True           # include area_of_law fixed effects (baseline dropped)
TRIM_LOW_PCTL = 0.0              # drop cases below this percentile (set None to disable)
TRIM_HIGH_PCTL = 0.99            # drop cases above this percentile (set None to disable)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
    )
}
BAILII_HEADERS = DEFAULT_HEADERS

HIGHCOURT_LIST_BASE = "https://www.bailii.org/ew/cases/EWHC/KB/"
HIGHCOURT_TOC_TEMPLATE = HIGHCOURT_LIST_BASE + "toc-{}.html"
HIGHCOURT_BASE_URL = "https://www.bailii.org"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CASE_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# Utilities
# -----------------------------

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def clean_text(s: str) -> str:
    s = re.sub(r"\r", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def parse_date_maybe(s: str) -> Optional[str]:
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

def compute_turnaround_days(case: Dict[str, Any]) -> Optional[int]:
    delta = days_between(case.get("hearing_start"), case.get("judgment_date"))
    if delta is None or delta < 0:
        return None
    return delta

def annotate_turnaround_days(cases: List[Dict[str, Any]]) -> None:
    for c in cases:
        c["turnaround_days"] = compute_turnaround_days(c)

def drop_negative_turnaround_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for c in cases:
        if c.get("turnaround_days") is None:
            continue
        if c["turnaround_days"] < 0:
            continue
        cleaned.append(c)
    return cleaned

def debug_log(msg: str) -> None:
    print(msg)

# -----------------------------
# HTTP helpers
# -----------------------------

def request_get(url: str, session: requests.Session, max_retries: int = 3, headers: Optional[Dict[str, str]] = None) -> Optional[requests.Response]:
    req_headers = headers or DEFAULT_HEADERS
    for attempt in range(max_retries):
        try:
            r = session.get(url, headers=req_headers, timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                return r
            if r.status_code in {429, 500, 502, 503, 504}:
                time.sleep(0.8 * (attempt + 1))
                continue
            return r
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    return None

# -----------------------------
# Judge parsing helpers
# -----------------------------

def canonicalize_judge_name(name: str) -> str:
    n = (name or "").strip()
    n = re.sub(r"\s*sitting as.*", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\s*acting as.*", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\s*the honourable\s+", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\s*\(.*?\)\s*$", "", n)
    n = re.sub(r"\s{2,}", " ", n)
    return n.strip().lower()

def is_valid_judge_name(name: str, min_len: int = 4) -> bool:
    if not name:
        return False
    n = name.strip()
    if len(n) < min_len:
        return False
    return bool(re.search(r"[A-Za-z]", n))

def dedup_judges_ordered(judges: List[str]) -> List[str]:
    from difflib import SequenceMatcher
    seen = []
    out = []
    for j in judges:
        canon = canonicalize_judge_name(j)
        if not canon:
            continue
        duplicate = False
        for c in seen:
            if c == canon or SequenceMatcher(None, c, canon).ratio() > 0.9:
                duplicate = True
                break
        if duplicate:
            continue
        seen.append(canon)
        out.append(j)
    return out

def filter_judges(judges: List[str]) -> List[str]:
    out = []
    seen = set()
    for j in judges or []:
        if not is_valid_judge_name(j):
            continue
        low = j.strip().lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(j)
    return out

# -----------------------------
# Parsing helpers
# -----------------------------

def extract_lines(soup: BeautifulSoup) -> List[str]:
    lines = [x.strip() for x in soup.get_text("\n").splitlines()]
    return [x for x in lines if x and x != "*"]

def soup_main_text(soup: BeautifulSoup) -> str:
    main = soup.find("main")
    node = main if main else soup
    return clean_text(node.get_text("\n"))

def highcourt_year_from_url(url: str) -> Optional[int]:
    m = re.search(r"/EWHC/KB/(\d{4})/", url, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def cache_path(case_url: str) -> Path:
    return CASE_JSON_DIR / f"{sha1(case_url)}.json"

def extract_panel_judges(soup: BeautifulSoup) -> List[str]:
    judges = []
    for p in soup.find_all("panel"):
        txt = p.get_text("\n")
        parts = [t.strip() for t in txt.split("\n") if t.strip()]
        if not parts:
            continue
        name = parts[0]
        name = re.sub(r"\s*\(.*?\)\s*$", "", name)
        name = re.sub(r"\s{2,}", " ", name).strip()
        if name and is_valid_judge_name(name):
            judges.append(name)
    return dedup_judges_ordered(judges)

def extract_judge_from_before(lines: List[str]) -> Optional[str]:
    judge_pat = re.compile(r"\b(?:MR|MRS|MS)\s+JUSTICE\s+[A-Z][A-Z]+(?:\s+[A-Z][A-Z]+)*\b", re.IGNORECASE)
    for i, ln in enumerate(lines[:200]):
        if not ln.lower().startswith("before"):
            continue
        m = judge_pat.search(ln)
        if m:
            j = m.group(0).title().replace("Mr ", "MR ").replace("Mrs ", "MRS ").replace("Ms ", "MS ")
            return j
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            m2 = judge_pat.search(nxt)
            if m2:
                j = m2.group(0).title().replace("Mr ", "MR ").replace("Mrs ", "MRS ").replace("Ms ", "MS ")
                return j
            if nxt.strip():
                return nxt.strip()
    return None

def extract_bailii_judges(lines: List[str]) -> List[str]:
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
                if is_valid_judge_name(cand):
                    judges.append(cand)
            break

    if not judges:
        for ln in lines[:150]:
            low = ln.lower()
            if len(ln) > 80:
                continue
            if "adjudged" in low or "judgment" in low:
                continue
            if re.match(r"^(the hon\.?\s+)?(lord|lady|mr|mrs|ms|miss|sir|dame)\s+.*(justice|judge)", ln, re.IGNORECASE):
                cand = ln.strip()
                if is_valid_judge_name(cand):
                    judges.append(cand)
            elif re.search(r"\b(his|her)\s+honour\s+judge\b", low):
                cand = ln.strip()
                if is_valid_judge_name(cand):
                    judges.append(cand)
            elif re.search(r"\bhhj\b", low) and "judge" in low:
                cand = ln.strip()
                if is_valid_judge_name(cand):
                    judges.append(cand)
        judges = judges[:3]

    return dedup_judges_ordered(judges)

def parse_highcourt_case(case_url: str, session: requests.Session) -> Optional[Dict[str, Any]]:
    def parse_hearing_start(text: str, lines: List[str]) -> Optional[str]:
        m = re.search(r"\bHearing\s+date[s]?:\s*([^\n]+)", text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"\bDate\s+of\s+hearing[s]?:\s*([^\n]+)", text, flags=re.IGNORECASE)
        if m:
            raw = re.split(r"[;,]", m.group(1))[0].strip()
            dt = parse_date_maybe(raw)
            if dt:
                return dt
        for ln in lines[:150]:
            m2 = re.search(r"\b[Hh]earing\s+date[s]?\s*[:-]\s*([A-Za-z0-9 ,/]+)", ln)
            if m2:
                dt = parse_date_maybe(m2.group(1))
                if dt:
                    return dt
        return None

    yr = highcourt_year_from_url(case_url)
    if yr is None or yr < MIN_CASE_YEAR or yr > CURRENT_YEAR:
        return None

    cpath = cache_path(case_url)
    if cpath.exists():
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
    flat_text = "\n".join(lines)
    panel_judges = extract_panel_judges(soup)

    title = clean_text(soup.title.get_text(" ")) if soup.title else None

    case_id = None
    case_no = None
    m_case_no = re.search(r"Case\s*No:\s*([A-Z]{1,5}-\d{4}-\d+)", flat_text, flags=re.IGNORECASE)
    if m_case_no:
        case_no = m_case_no.group(1).strip()
    citation_pattern = r"\[\d{4}\]\s*EWHC\s*\d+\s*\([A-Za-z]{2,5}\)"
    for ln in lines[:200]:
        m = re.search(citation_pattern, ln)
        if m:
            case_id = m.group(0)
            break
    if not case_id:
        m = re.search(citation_pattern, flat_text)
        if m:
            case_id = m.group(0)
    if not case_id and title:
        m = re.search(citation_pattern, title)
        if m:
            case_id = m.group(0)
    if not case_id:
        parts = case_url.rstrip("/").split("/")
        if len(parts) >= 3:
            year = parts[-2] if parts[-2].isdigit() else None
            num = parts[-1].replace(".html", "")
            if year and num:
                case_id = f"[{year}] EWHC {num} (KB)"

    judgment_date = None
    hearing_start = None
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
    if not judgment_date:
        m = re.search(r"\bDate:\s*([0-3]?\d/[01]?\d/\d{4})\b", flat_text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"\bDate:\s*([0-3]?\d\s+[A-Za-z]+\s+\d{4})\b", flat_text, flags=re.IGNORECASE)
        if m:
            judgment_date = parse_date_maybe(m.group(1))

    if not hearing_start:
        hearing_start = parse_hearing_start(flat_text, lines)

    justices = panel_judges
    if not justices:
        judge_from_before = extract_judge_from_before(lines)
        if judge_from_before and is_valid_judge_name(judge_from_before):
            justices = [judge_from_before]
        else:
            justices = extract_bailii_judges(lines)

    justices = filter_judges(justices)
    if not justices:
        return None

    text_full = soup_main_text(soup)
    area_of_law = "High Court (King's Bench)"
    for ln in lines[:120]:
        if len(ln) > 80:
            continue
        if re.search(r"\bdivision\b", ln, flags=re.IGNORECASE) or re.search(r"\bcourt\b", ln, flags=re.IGNORECASE):
            area_of_law = ln.strip()
            break

    if not hearing_start or not judgment_date:
        return None

    out = {
        "case_url": case_url,
        "judgment_url": case_url,
        "case_id": case_id or "",
        "case_no": case_no or "",
        "title": title or "",
        "area_of_law": area_of_law,
        "issue": "",
        "facts": "",
        "date_of_issue": None,
        "judgment_date": judgment_date,
        "hearing_start": hearing_start,
        "hearing_end": None,
        "justices": justices[:1],  # keep primary judge only for speed model
        "opinion_author": "",
        "text_full": text_full[:2_000_000],
        "analysis_text": text_full[:2_000_000],
        "source_urls": {"case": case_url, "judgment": case_url},
        "_parse_version": CASE_PARSE_VERSION,
    }

    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out

# -----------------------------
# Crawling
# -----------------------------

def list_highcourt_case_urls(session: requests.Session) -> List[str]:
    letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    urls: List[str] = []
    seen = set()
    for letter in letters:
        toc_url = HIGHCOURT_TOC_TEMPLATE.format(letter)
        r = request_get(toc_url, session, headers=BAILII_HEADERS)
        if r is None or r.status_code != 200:
            continue
        html = r.text or ""
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if "/ew/cases/EWHC/KB/" not in href:
                continue
            if not href.lower().endswith(".html"):
                continue
            full = href
            if href.startswith("/"):
                full = HIGHCOURT_BASE_URL.rstrip("/") + href
            year = highcourt_year_from_url(full)
            if year is None or year < MIN_CASE_YEAR or year > CURRENT_YEAR:
                continue
            if full in seen:
                continue
            seen.add(full)
            urls.append(full)
        time.sleep(REQUEST_DELAY_SEC)

    urls.sort()
    return urls

def load_cases_cache(path: Path) -> List[Dict[str, Any]]:
    # show where we are trying to load from (helps when running in different cwd)
    print(f"[cache] attempting load from {path} (dir={path.parent})")
    if not path.exists():
        print(f"[cache] miss: file not found at {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            if data:
                found_version = data[0].get("_parse_version")
                if found_version != CASE_PARSE_VERSION:
                    print(f"[cache] parse_version mismatch (found {found_version}, expected {CASE_PARSE_VERSION}); ignoring cache")
                    return []
            print(f"[cache] loaded {len(data)} cached cases from {path}")
            return data
        else:
            print(f"[cache] unexpected cache format (type={type(data).__name__}); ignoring")
    except Exception as exc:
        print(f"[cache] failed to load {path}: {exc}")
    return []

def save_cases_cache(path: Path, cases: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def crawl_cases(n_cases: int, session: requests.Session, cache_path: Optional[Path] = None, seed_cases: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = list(seed_cases) if seed_cases else []
    seen: set = {c.get("case_url") for c in found if c.get("case_url")}
    if len(found) >= n_cases:
        return found[:n_cases]

    last_saved = len(found)

    def handle_urls(urls: List[str], progress_desc: Optional[str] = None) -> bool:
        nonlocal last_saved
        iterator = urls
        if progress_desc:
            iterator = tqdm(urls, desc=progress_desc)
        for u in iterator:
            if u in seen:
                continue
            seen.add(u)

            case = parse_highcourt_case(u, session)
            time.sleep(REQUEST_DELAY_SEC)

            if case is None:
                continue

            case["justices"] = filter_judges(case.get("justices", []))
            found.append(case)

            if cache_path and (len(found) - last_saved) >= CACHE_SAVE_EVERY:
                save_cases_cache(cache_path, found)
                last_saved = len(found)

            if len(found) >= n_cases:
                if cache_path:
                    save_cases_cache(cache_path, found)
                return True
        return False

    urls = list_highcourt_case_urls(session)
    if urls:
        handle_urls(urls, progress_desc="Parsing High Court cases")
    if cache_path:
        save_cases_cache(cache_path, found)
    return found

# -----------------------------
# Speed analysis
# -----------------------------

def year_from_iso(date_iso: Optional[str]) -> Optional[int]:
    if not date_iso:
        return None
    try:
        return int(date_iso.split("-")[0])
    except Exception:
        return None

def build_speed_dataset(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for c in cases:
        tdays = c.get("turnaround_days")
        if tdays is None:
            continue
        judge = None
        judges = filter_judges(c.get("justices", []) or [])
        if judges:
            judge = judges[0]
        if not judge:
            continue
        rows.append({
            "case_id": c.get("case_id"),
            "title": c.get("title"),
            "case_url": c.get("case_url"),
            "judge": judge,
            "turnaround_days": tdays,
            "hearing_start": c.get("hearing_start"),
            "judgment_date": c.get("judgment_date"),
            "hearing_year": year_from_iso(c.get("hearing_start")),
            "judgment_year": year_from_iso(c.get("judgment_date")),
            "area_of_law": c.get("area_of_law"),
        })
    return rows

def trim_turnaround(rows: List[Dict[str, Any]], p_low: Optional[float], p_high: Optional[float]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Remove rows with turnaround_days outside [p_low, p_high] percentiles.
    """
    if rows is None or len(rows) == 0 or p_high is None:
        return rows, {}
    vals = np.array([r["turnaround_days"] for r in rows], dtype=float)
    low = np.percentile(vals, p_low * 100) if p_low is not None else vals.min()
    high = np.percentile(vals, p_high * 100)
    kept = [r for r in rows if low <= r["turnaround_days"] <= high]
    dropped = len(rows) - len(kept)
    return kept, {"low": float(low), "high": float(high), "dropped": dropped}

def winsorize_turnaround(rows: List[Dict[str, Any]], p_low: float, p_high: float) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Clip turnaround_days at given percentiles. Returns (rows, stats).
    """
    if rows is None or len(rows) == 0 or p_high is None:
        return rows, {}
    vals = np.array([r["turnaround_days"] for r in rows], dtype=float)
    lo = np.percentile(vals, p_low * 100) if p_low and p_low > 0 else vals.min()
    hi = np.percentile(vals, p_high * 100) if p_high and p_high < 1.0 else vals.max()
    for r in rows:
        v = r["turnaround_days"]
        if v < lo:
            r["turnaround_days"] = float(lo)
        elif v > hi:
            r["turnaround_days"] = float(hi)
    return rows, {"low": float(lo), "high": float(hi)}

def summarize_turnaround(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    vals = np.array([r["turnaround_days"] for r in rows], dtype=float)
    if vals.size == 0:
        return {}
    return {
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }

def build_design_matrix(
    rows: List[Dict[str, Any]],
    target_key: str = "turnaround_days",
    include_judge: bool = True,
    include_judgment_year: bool = True,
    include_hearing_year: bool = False,
    include_area: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    y = np.array([r[target_key] for r in rows], dtype=float)

    cols = []
    col_names = []
    judge_names = []
    baseline_judge = None

    if include_judge:
        judge_names = sorted({r["judge"] for r in rows})
        if judge_names:
            baseline_judge = judge_names[0]
            for j in judge_names[1:]:
                col = np.array([1.0 if r["judge"] == j else 0.0 for r in rows], dtype=float)
                cols.append(col)
                col_names.append(f"judge[{j}]")

    judgment_year_names = []
    baseline_judgment_year = None
    if include_judgment_year:
        judgment_year_names = sorted({r["judgment_year"] for r in rows if r.get("judgment_year") is not None})
        if judgment_year_names:
            baseline_judgment_year = judgment_year_names[0]
            for yv in judgment_year_names[1:]:
                col = np.array([1.0 if r.get("judgment_year") == yv else 0.0 for r in rows], dtype=float)
                cols.append(col)
                col_names.append(f"judgment_year[{yv}]")

    hearing_year_names = []
    baseline_hearing_year = None
    if include_hearing_year:
        hearing_year_names = sorted({r["hearing_year"] for r in rows if r.get("hearing_year") is not None})
        if hearing_year_names:
            baseline_hearing_year = hearing_year_names[0]
            for yv in hearing_year_names[1:]:
                col = np.array([1.0 if r.get("hearing_year") == yv else 0.0 for r in rows], dtype=float)
                cols.append(col)
                col_names.append(f"hearing_year[{yv}]")

    area_names = []
    baseline_area = None
    if include_area:
        area_names = sorted({r["area_of_law"] for r in rows if r.get("area_of_law")})
        if area_names:
            baseline_area = area_names[0]
            for a in area_names[1:]:
                col = np.array([1.0 if r.get("area_of_law") == a else 0.0 for r in rows], dtype=float)
                cols.append(col)
                col_names.append(f"area[{a}]")

    if cols:
        X = np.column_stack([np.ones(len(rows), dtype=float)] + cols)
    else:
        X = np.ones((len(rows), 1), dtype=float)

    meta = {
        "col_names": col_names,
        "baseline_judge": baseline_judge,
        "judge_names": judge_names,
        "judgment_year_names": judgment_year_names,
        "baseline_judgment_year": baseline_judgment_year,
        "hearing_year_names": hearing_year_names,
        "baseline_hearing_year": baseline_hearing_year,
        "area_names": area_names,
        "baseline_area": baseline_area,
        "n_params": X.shape[1],
        "n_judge_params": len(judge_names) - 1 if include_judge else 0,
    }
    return X, y, meta

def fit_ols(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    ssr = float(resid @ resid)
    sst = float(((y - y.mean()) @ (y - y.mean()))) if y.size else 0.0
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    df_resid = max(0, len(y) - X.shape[1])
    return {
        "beta": beta,
        "y_hat": y_hat,
        "resid": resid,
        "ssr": ssr,
        "sst": sst,
        "r2": r2,
        "df_resid": df_resid,
    }

def f_test_judge_effects(rows: List[Dict[str, Any]], target_key: str = "turnaround_days") -> Dict[str, Any]:
    X_full, y, meta_full = build_design_matrix(
        rows,
        target_key=target_key,
        include_judge=True,
        include_judgment_year=INCLUDE_JUDGMENT_YEAR,
        include_hearing_year=INCLUDE_HEARING_YEAR,
        include_area=INCLUDE_AREA_FE
    )
    full = fit_ols(X_full, y)

    X_restricted, _, _ = build_design_matrix(
        rows,
        target_key=target_key,
        include_judge=False,
        include_judgment_year=INCLUDE_JUDGMENT_YEAR,
        include_hearing_year=INCLUDE_HEARING_YEAR,
        include_area=INCLUDE_AREA_FE
    )
    restricted = fit_ols(X_restricted, y)

    q = meta_full["n_judge_params"]
    if q <= 0:
        return {"n_judges": len(meta_full.get("judge_names", [])), "f_stat": None, "p_value": None}

    df_den = full["df_resid"]
    if df_den <= 0:
        return {"n_judges": len(meta_full.get("judge_names", [])), "f_stat": None, "p_value": None}

    f_stat = ((restricted["ssr"] - full["ssr"]) / q) / (full["ssr"] / df_den)
    p_val = None
    if f_dist is not None and math.isfinite(f_stat):
        try:
            p_val = float(f_dist.sf(f_stat, q, df_den))
        except Exception:
            p_val = None
    return {
        "n_judges": len(meta_full.get("judge_names", [])),
        "f_stat": float(f_stat) if math.isfinite(f_stat) else None,
        "p_value": p_val,
        "df_num": q,
        "df_den": df_den,
        "r2": full["r2"],
        "adj_r2": 1 - (1 - full["r2"]) * (len(y) - 1) / max(1, df_den),
        "baselines": {
            "judge": meta_full.get("baseline_judge"),
            "judgment_year": meta_full.get("baseline_judgment_year"),
            "hearing_year": meta_full.get("baseline_hearing_year"),
            "area": meta_full.get("baseline_area"),
        },
        "beta": full["beta"].tolist(),
        "col_names": meta_full.get("col_names", []),
    }

def judge_effects(rows: List[Dict[str, Any]], f_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    X_full, y, meta = build_design_matrix(
        rows,
        target_key="turnaround_days",
        include_judge=True,
        include_judgment_year=INCLUDE_JUDGMENT_YEAR,
        include_hearing_year=INCLUDE_HEARING_YEAR,
        include_area=INCLUDE_AREA_FE
    )
    beta = np.array(f_result.get("beta"))
    if beta.size == 0:
        return []
    col_names = meta.get("col_names", [])

    # Map beta coefficients back to judges; baseline judge effect = 0
    effects = {meta.get("baseline_judge"): 0.0}
    for name, coeff in zip(col_names, beta[1:]):  # skip intercept
        if name.startswith("judge["):
            j = name[len("judge["):-1]
            effects[j] = float(coeff)

    # Aggregate observed turnaround by judge
    by_judge = defaultdict(list)
    for r in rows:
        by_judge[r["judge"]].append(r["turnaround_days"])

    out = []
    for j, vals in by_judge.items():
        eff = effects.get(j, 0.0)
        arr = np.array(vals, dtype=float)
        out.append({
            "judge": j,
            "n_cases": int(arr.size),
            "mean_turnaround": float(arr.mean()),
            "median_turnaround": float(np.median(arr)),
            "effect_days": float(eff),
            "is_baseline": j == meta.get("baseline_judge"),
        })

    out.sort(key=lambda x: x["effect_days"])
    return out

def dispersion_stats(effects: List[Dict[str, Any]]) -> Dict[str, float]:
    vals = np.array([e["effect_days"] for e in effects], dtype=float)
    if vals.size == 0:
        return {}
    return {
        "std": float(np.std(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
        "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        "range": float(np.max(vals) - np.min(vals)),
    }

# -----------------------------
# Plotting helpers
# -----------------------------

def plot_turnaround(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    out = {}
    vals = np.array([r["turnaround_days"] for r in rows], dtype=float)
    if vals.size == 0:
        return out

    def _save(fig, name: str):
        path = PLOTS_DIR / name
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        out[name] = str(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=60, color="#4c78a8", alpha=0.85)
    ax.set_title("Turnaround (days)")
    ax.set_xlabel("days")
    ax.set_ylabel("count")
    ax.axvline(vals.mean(), color="red", linestyle="--", label=f"mean={vals.mean():.1f}")
    ax.axvline(np.median(vals), color="green", linestyle="--", label=f"median={np.median(vals):.1f}")
    ax.legend()
    _save(fig, "turnaround_hist.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log1p(vals), bins=60, color="#f58518", alpha=0.85)
    ax.set_title("Turnaround log1p(days)")
    ax.set_xlabel("log1p(days)")
    ax.set_ylabel("count")
    _save(fig, "turnaround_hist_log.png")

    return out

def plot_judge_effects(effects: List[Dict[str, Any]], top_n: int = 25) -> Dict[str, str]:
    out = {}
    if not effects:
        return out
    sorted_eff = sorted(effects, key=lambda x: x["effect_days"])
    sample = sorted_eff[:top_n] + sorted_eff[-top_n:] if len(sorted_eff) > top_n else sorted_eff
    labels = [e["judge"] for e in sample]
    vals = [e["effect_days"] for e in sample]

    fig, ax = plt.subplots(figsize=(8, max(6, 0.25 * len(sample))))
    ax.barh(range(len(sample)), vals, color="#54a24b")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(range(len(sample)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Judge effect (days vs baseline; negative = faster)")
    ax.set_title("Judge fixed effects (sample)")
    fig.tight_layout()
    path = PLOTS_DIR / "judge_effects_bar.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    out["judge_effects_bar.png"] = str(path)

    # Boxplot for overall spread
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.boxplot(vals, vert=True, showfliers=True)
    ax.set_ylabel("Judge effect (days)")
    ax.set_title("Judge effects spread (sample)")
    fig.tight_layout()
    path2 = PLOTS_DIR / "judge_effects_box.png"
    fig.savefig(path2, dpi=180)
    plt.close(fig)
    out["judge_effects_box.png"] = str(path2)

    return out

# -----------------------------
# Reporting helpers
# -----------------------------

def print_readable_summary(
    n_cases: int,
    f_res: Dict[str, Any],
    disp: Dict[str, Any],
    t_summary: Dict[str, Any],
    log_res: Optional[Dict[str, Any]] = None
) -> None:
    """
    Emit a compact, human-readable summary aligned to the review questions.
    """
    fstat = f_res.get("f_stat")
    pval = f_res.get("p_value")
    df_num = f_res.get("df_num")
    df_den = f_res.get("df_den")
    decision = None
    if pval is not None:
        if pval < 0.01:
            decision = "Reject H0 at 1% (strong evidence of judge effects)."
        elif pval < 0.05:
            decision = "Reject H0 at 5% (judge effects significant)."
        elif pval < 0.10:
            decision = "Reject H0 at 10% (weak evidence)."
        else:
            decision = "Fail to reject H0 (no joint significance)."

    print("\n=== READABLE SUMMARY ===")
    print("Speed measure:")
    print(f"  hearing -> judgment days | cases with valid speed: {n_cases}")
    print(f"  mean / median turnaround: {t_summary.get('mean'):.1f} / {t_summary.get('median'):.1f} days")
    print("\nModel:")
    print("  speed_ij = alpha + gamma_judge + controls (judgment year) + epsilon_ij")
    print("\nTest: Are judge fixed effects jointly significant?")
    if fstat is not None and df_num is not None and df_den is not None:
        print(f"  F({df_num}, {df_den}) = {fstat:.3f}   p = {pval:.4f}   -> {decision}")
    else:
        print("  F-test not available")
    if f_res.get("r2") is not None:
        print(f"  Model fit: R^2 = {f_res.get('r2'):.3f}   adj R^2 = {f_res.get('adj_r2'):.3f}")
    print("\nDispersion of judge effects (days relative to baseline; negative = faster):")
    span = (disp.get("p90", 0) - disp.get("p10", 0)) if disp else 0
    print(f"  Std dev: {disp.get('std'):.2f}")
    print(f"  P10 -> P90: {disp.get('p10'):.1f} -> {disp.get('p90'):.1f} (span {span:.1f})")
    print(f"  IQR: {disp.get('iqr'):.1f}   Range: {disp.get('range'):.1f}")
    print("  Interpretation: gamma_judge < 0 = faster than baseline; > 0 = slower.")
    if log_res:
        lf = log_res.get("f_test", {})
        print("\nLog(turnaround+1) robustness:")
        if lf.get("stat") is not None:
            print(f"  F({lf.get('df_num')}, {lf.get('df_den')}) = {lf.get('stat'):.3f}   p = {lf.get('p_value'):.4f}")
            if log_res.get("r2") is not None:
                print(f"  R^2 = {log_res.get('r2'):.3f}   adj R^2 = {log_res.get('adj_r2'):.3f}")
        else:
            print("  Not available")
    print("========================\n")

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    session = requests.Session()

    print("=== CASE INGESTION ===")
    seed_cases = load_cases_cache(CASES_CACHE_PATH)
    if seed_cases:
        print(f"Loaded {len(seed_cases)} cached cases from {CASES_CACHE_PATH}")

    if len(seed_cases) >= MAX_CASES:
        cases = seed_cases[:MAX_CASES]
        print("Using cached cases; skipping crawl.")
    else:
        print("Crawling cases (with judgment dates)...")
        cases = crawl_cases(MAX_CASES, session, cache_path=CASES_CACHE_PATH, seed_cases=seed_cases)
        print(f"Collected {len(cases)} cases.")

    annotate_turnaround_days(cases)
    cases = drop_negative_turnaround_cases(cases)
    if len(cases) < 30:
        print("Too few cases with valid turnaround times to run analysis.")
        return

    print("=== SPEED DATASET ===")
    speed_rows = build_speed_dataset(cases)
    print(f"Cases with speed + judge: {len(speed_rows)}")

    trim_info = {}
    if TRIM_HIGH_PCTL is not None:
        speed_rows, trim_info = trim_turnaround(speed_rows, TRIM_LOW_PCTL, TRIM_HIGH_PCTL)
        print(f"Trimmed turnaround outside [{trim_info.get('low'):.1f}, {trim_info.get('high'):.1f}] (pctl {TRIM_LOW_PCTL*100:.1f}-{TRIM_HIGH_PCTL*100:.1f}); dropped {trim_info.get('dropped',0)} cases")

    winsor_info = {}
    if WINSOR_HIGH_PCTL is not None:
        speed_rows, winsor_info = winsorize_turnaround(speed_rows, WINSOR_LOW_PCTL, WINSOR_HIGH_PCTL)
        if winsor_info:
            print(f"Winsorized turnaround at [{winsor_info.get('low'):.1f}, {winsor_info.get('high'):.1f}] (pctl {WINSOR_LOW_PCTL*100:.1f}-{WINSOR_HIGH_PCTL*100:.1f})")

    print("=== MODEL: judge fixed effects on turnaround ===")
    f_res = f_test_judge_effects(speed_rows)
    effects = judge_effects(speed_rows, f_res)
    disp = dispersion_stats(effects)
    t_summary = summarize_turnaround(speed_rows)
    plot_paths = {}
    plot_paths.update(plot_turnaround(speed_rows))
    plot_paths.update(plot_judge_effects(effects))

    # Optional log model
    log_model_res = None
    if USE_LOG_MODEL:
        rows_log = []
        for r in speed_rows:
            rlog = dict(r)
            rlog["turnaround_log"] = math.log1p(r["turnaround_days"])
            rows_log.append(rlog)
        log_f = f_test_judge_effects(rows_log, target_key="turnaround_log")
        log_model_res = {
            "f_test": {
                "stat": log_f.get("f_stat"),
                "p_value": log_f.get("p_value"),
                "df_num": log_f.get("df_num"),
                "df_den": log_f.get("df_den"),
            },
            "r2": log_f.get("r2"),
            "adj_r2": log_f.get("adj_r2"),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CASES_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(speed_rows, f, ensure_ascii=False, indent=2)
    with open(JUDGE_EFFECTS_PATH, "w", encoding="utf-8") as f:
        json.dump(effects, f, ensure_ascii=False, indent=2)
    with open(SPEED_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_cases": len(speed_rows),
            "n_judges": f_res.get("n_judges"),
            "controls": [
                c for c in [
                    "judgment_year" if INCLUDE_JUDGMENT_YEAR else None,
                    "hearing_year" if INCLUDE_HEARING_YEAR else None,
                    "area_of_law" if INCLUDE_AREA_FE else None
                ] if c
            ],
            "turnaround_summary": t_summary,
            "f_test": {
                "stat": f_res.get("f_stat"),
                "p_value": f_res.get("p_value"),
                "df_num": f_res.get("df_num"),
                "df_den": f_res.get("df_den"),
            },
            "r2": f_res.get("r2"),
            "adj_r2": f_res.get("adj_r2"),
            "dispersion": disp,
            "baselines": f_res.get("baselines"),
            "trim": trim_info,
            "winsor": winsor_info,
            "log_model": log_model_res,
            "plots": plot_paths,
        }, f, ensure_ascii=False, indent=2)

    print("=== RESULTS ===")
    print(json.dumps({
        "cases": len(speed_rows),
        "judges": f_res.get("n_judges"),
        "f_stat": f_res.get("f_stat"),
        "p_value": f_res.get("p_value"),
        "dispersion_std": disp.get("std"),
        "p90_minus_p10": (disp.get("p90", 0) - disp.get("p10", 0)) if disp else None,
    }, indent=2))
    print(f"Saved: {CASES_OUTPUT_PATH}, {JUDGE_EFFECTS_PATH}, {SPEED_RESULTS_PATH}")
    print_readable_summary(len(speed_rows), f_res, disp, t_summary, log_model_res)
    if plot_paths:
        print("Saved plots:")
        for name, path in plot_paths.items():
            print(f"  - {name}: {path}")

if __name__ == "__main__":
    main()
