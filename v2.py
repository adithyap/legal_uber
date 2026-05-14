#!/usr/bin/env python3
"""
BAILII EWHC KB/QB judge reassignment analysis.

Primary question:
    If cases had been assigned differently to judges, would the court have
    moved faster?

The primary speed metric is reserved_days = judgment_date - hearing_end.
The pipeline keeps BAILII as the source of record, validates parsed fields
before modelling, estimates regularized judge speed effects, and produces a
static dashboard that can recompute constrained reassignment scenarios in the
browser.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import numpy as np
import requests
from dateutil import parser as dateparser

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional parser dependency
    BeautifulSoup = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is a convenience only
    def tqdm(items: Iterable[Any], **_: Any) -> Iterable[Any]:
        return items


# -----------------------------
# Config
# -----------------------------

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CURRENT_YEAR = datetime.now(timezone.utc).year
MIN_CASE_YEAR = CURRENT_YEAR - 25
REQUEST_DELAY_SEC = 0.2
HTTP_TIMEOUT = 30
CACHE_SAVE_EVERY = 25

CASE_PARSE_VERSION = 20
OUTPUT_DIR = Path("data")
CASES_RAW_PATH = OUTPUT_DIR / "cases_raw.json"
CASES_OUTPUT_PATH = OUTPUT_DIR / "cases_speed.json"
JUDGE_EFFECTS_PATH = OUTPUT_DIR / "judge_effects.json"
SPEED_RESULTS_PATH = OUTPUT_DIR / "speed_results.json"
ASSIGNMENT_RESULTS_PATH = OUTPUT_DIR / "assignment_results.json"
DASHBOARD_DATA_PATH = OUTPUT_DIR / "dashboard_data.json"
CRAWL_REJECTIONS_PATH = OUTPUT_DIR / "crawl_rejections.json"
JUDGE_ALIASES_PATH = OUTPUT_DIR / "judge_aliases.json"
INDEX_PATH = OUTPUT_DIR / "index.html"

BAILII_BASE_URL = "https://www.bailii.org"
BAILII_LIST_BASE = "https://www.bailii.org/ew/cases/EWHC/{division}/"
BAILII_TOC_TEMPLATE = BAILII_LIST_BASE + "toc-{letter}.html"
COURT_DIVISIONS = ("KB", "QB")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
    )
}

DEFAULT_MIN_JUDGE_CASES = 5
DEFAULT_CAPACITY_SLACK = 0.10
DEFAULT_TIME_BUCKET = "year"
RIDGE_ALPHA = 8.0


class BailiiBlockedError(RuntimeError):
    """Raised when BAILII returns a bot-check or other blocking page."""


class HtmlTextAndLinkParser(HTMLParser):
    """Small dependency-free parser for BAILII HTML pages."""

    BLOCK_TAGS = {
        "address",
        "article",
        "blockquote",
        "br",
        "center",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: List[str] = []
        self.text_parts: List[str] = []
        self.title_parts: List[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag == "a":
            for key, val in attrs:
                if key.lower() == "href" and val:
                    self.links.append(val.strip())
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not data:
            return
        self.text_parts.append(data)
        if self._in_title:
            self.title_parts.append(data)


def clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = value.replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def html_to_text_and_links(raw_html: str) -> Tuple[str, str, List[str]]:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(raw_html or "", "html.parser")
        title = clean_text(soup.title.get_text(" ")) if soup.title else ""
        text = clean_text(soup.get_text("\n"))
        links = [str(a.get("href")).strip() for a in soup.find_all("a", href=True)]
        return title, text, links

    parser = HtmlTextAndLinkParser()
    parser.feed(raw_html or "")
    title = clean_text(" ".join(parser.title_parts))
    text = clean_text("".join(parser.text_parts))
    return title, text, parser.links


def text_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip() and ln.strip() != "*"]


def sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def parse_date_maybe(value: str) -> Optional[str]:
    if not value:
        return None
    try:
        parsed = dateparser.parse(value, dayfirst=True, fuzzy=True)
    except Exception:
        return None
    if parsed is None:
        return None
    return parsed.date().isoformat()


def iso_to_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return dateparser.parse(value).date()
    except Exception:
        return None


def days_between(start_iso: Optional[str], end_iso: Optional[str]) -> Optional[int]:
    start = iso_to_date(start_iso)
    end = iso_to_date(end_iso)
    if not start or not end:
        return None
    return int((end - start).days)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


# -----------------------------
# BAILII crawling
# -----------------------------


def is_bailii_blocked(html_text: str) -> bool:
    lower = (html_text or "").lower()
    return (
        "making sure you&#39;re not a bot" in lower
        or "making sure you're not a bot" in lower
        or "/.within.website/" in lower
        or ("bot" in lower and "noindex,nofollow" in lower)
        or "captcha" in lower
    )


def request_get(url: str, session: requests.Session, max_retries: int = 3) -> Optional[requests.Response]:
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=DEFAULT_HEADERS, timeout=HTTP_TIMEOUT)
        except Exception:
            time.sleep(0.8 * (attempt + 1))
            continue

        if response.status_code == 200:
            if is_bailii_blocked(response.text):
                raise BailiiBlockedError(
                    "BAILII returned a bot-check page. The crawler will not bypass it; "
                    "retry from an allowed environment or use the existing cache."
                )
            return response

        if response.status_code in {403, 429}:
            raise BailiiBlockedError(f"BAILII blocked the request to {url} with HTTP {response.status_code}.")

        if response.status_code in {500, 502, 503, 504}:
            time.sleep(0.8 * (attempt + 1))
            continue

        return response
    return None


def case_division_from_url(url: str) -> Optional[str]:
    match = re.search(r"/EWHC/(KB|QB)/\d{4}/", url or "", flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def case_year_from_url(url: str) -> Optional[int]:
    match = re.search(r"/EWHC/(?:KB|QB)/(\d{4})/", url or "", flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def list_bailii_case_urls(session: requests.Session, divisions: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    urls: List[str] = []

    for division in divisions:
        division = division.upper()
        for letter_ord in range(ord("A"), ord("Z") + 1):
            letter = chr(letter_ord)
            toc_url = BAILII_TOC_TEMPLATE.format(division=division, letter=letter)
            response = request_get(toc_url, session)
            if response is None or response.status_code != 200:
                continue

            _, _, links = html_to_text_and_links(response.text)
            for href in links:
                full = urljoin(BAILII_BASE_URL, href)
                if f"/ew/cases/EWHC/{division}/" not in full:
                    continue
                if not full.lower().endswith(".html"):
                    continue
                year = case_year_from_url(full)
                if year is None or year < MIN_CASE_YEAR or year > CURRENT_YEAR:
                    continue
                if full not in seen:
                    seen.add(full)
                    urls.append(full)
            time.sleep(REQUEST_DELAY_SEC)

    def sort_key(url: str) -> Tuple[str, int, int, str]:
        div = case_division_from_url(url) or ""
        year = case_year_from_url(url) or 0
        num_match = re.search(r"/(\d+)\.html$", url)
        num = int(num_match.group(1)) if num_match else 0
        return div, year, num, url

    return sorted(urls, key=sort_key)


# -----------------------------
# Parsing
# -----------------------------


MONTH_ALIASES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
MONTH_PATTERN = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"


def normalize_date_text(value: str) -> str:
    out = html.unescape(value or "")
    out = out.replace("\x96", "-").replace("\x97", "-")
    out = re.sub(r"[\u2010-\u2015]", "-", out)
    out = out.replace("&", " and ")
    out = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def extract_hearing_date_raw(lines: List[str]) -> Optional[str]:
    for idx, line in enumerate(lines[:350]):
        if re.search(r"\b(?:hearing date|hearing dates|date of hearing|dates of hearing)\b", line, flags=re.IGNORECASE):
            raw = line.strip()
            if re.search(r":\s*$", raw) and idx + 1 < len(lines):
                raw = f"{raw} {lines[idx + 1].strip()}"
            return raw[:500]
    return None


def _year_for_month_group(body: str, month_end: int, next_month_start: int, final_year: Optional[int]) -> Optional[int]:
    following = body[month_end:next_month_start]
    match = re.search(r"\b(19\d{2}|20\d{2})\b", following)
    if match:
        return int(match.group(1))
    return final_year


def parse_hearing_dates(raw: Optional[str], judgment_date_iso: Optional[str] = None) -> Tuple[Optional[str], Optional[str], List[str]]:
    warnings: List[str] = []
    if not raw:
        return None, None, ["missing_hearing_date_raw"]

    body = normalize_date_text(raw)
    body = re.sub(
        r"^\s*(?:hearing date|hearing dates|date of hearing|dates of hearing)\s*:?\s*",
        "",
        body,
        flags=re.IGNORECASE,
    )
    clauses = [part.strip() for part in body.split(";") if part.strip()]
    if len(clauses) > 1:
        date_clauses = []
        for clause in clauses:
            has_date = re.search(MONTH_PATTERN, clause, flags=re.IGNORECASE) or re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", clause)
            is_written_only = re.search(r"\b(?:further\s+)?written\s+submissions\b|\bpost-hearing\s+submissions\b", clause, flags=re.IGNORECASE)
            if has_date and not is_written_only:
                date_clauses.append(clause)
        if date_clauses:
            body = "; ".join(date_clauses)

    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", body)]
    final_year = years[-1] if years else None
    judgment_dt = iso_to_date(judgment_date_iso)
    if final_year is None and judgment_dt is not None:
        final_year = judgment_dt.year
        warnings.append("hearing_year_inferred_from_judgment_year")

    dates: List[date] = []
    month_matches = list(re.finditer(MONTH_PATTERN, body, flags=re.IGNORECASE))
    prev_end = 0
    for idx, match in enumerate(month_matches):
        segment = body[prev_end:match.start()]
        next_start = month_matches[idx + 1].start() if idx + 1 < len(month_matches) else len(body)
        year = _year_for_month_group(body, match.end(), next_start, final_year)
        month_key = match.group(0).lower().rstrip(".")
        month_num = MONTH_ALIASES.get(month_key)
        if year is None or month_num is None:
            prev_end = match.end()
            continue

        days: set[int] = set()
        for start_s, end_s in re.findall(r"\b(\d{1,2})\s*-\s*(\d{1,2})\b", segment):
            start_day, end_day = int(start_s), int(end_s)
            if 1 <= start_day <= 31:
                days.add(start_day)
            if 1 <= end_day <= 31:
                days.add(end_day)
        for day_s in re.findall(r"\b(\d{1,2})\b", segment):
            day = int(day_s)
            if 1 <= day <= 31:
                days.add(day)

        for day in days:
            try:
                dates.append(date(year, month_num, day))
            except ValueError:
                warnings.append(f"invalid_hearing_date_component:{year}-{month_num:02d}-{day:02d}")
        prev_end = match.end()

    if not dates:
        for match in re.finditer(rf"\b\d{{1,2}}\s+{MONTH_PATTERN}\s+\d{{4}}\b", body, flags=re.IGNORECASE):
            parsed = parse_date_maybe(match.group(0))
            parsed_dt = iso_to_date(parsed)
            if parsed_dt:
                dates.append(parsed_dt)

    if not dates:
        for match in re.finditer(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", body):
            parsed = parse_date_maybe(match.group(0))
            parsed_dt = iso_to_date(parsed)
            if parsed_dt:
                dates.append(parsed_dt)

    if judgment_dt is not None and dates:
        adjusted: List[date] = []
        for dt in dates:
            if dt > judgment_dt and "hearing_year_inferred_from_judgment_year" in warnings:
                try:
                    candidate = date(dt.year - 1, dt.month, dt.day)
                except ValueError:
                    candidate = dt
                adjusted.append(candidate if candidate <= judgment_dt else dt)
            else:
                adjusted.append(dt)
        dates = adjusted

    dates = sorted(set(dates))
    if not dates:
        warnings.append("unable_to_parse_hearing_dates")
        return None, None, warnings

    start, end = dates[0], dates[-1]
    if judgment_dt is not None and end > judgment_dt:
        warnings.append("hearing_end_after_judgment")

    return start.isoformat(), end.isoformat(), warnings


def extract_judgment_date(lines: List[str], title: str, existing: Optional[str] = None) -> Optional[str]:
    title_match = re.search(r"\((\d{1,2}\s+[A-Za-z]+\s+\d{4})\)", title or "")
    if title_match:
        parsed = parse_date_maybe(title_match.group(1))
        if parsed:
            return parsed

    if existing:
        parsed = parse_date_maybe(existing)
        if parsed:
            return parsed

    for line in lines[:160]:
        if re.search(r"\bhearing date", line, flags=re.IGNORECASE):
            continue
        match = re.search(r"\bDate\s*:?\s*([0-3]?\d(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}|[0-3]?\d/[01]?\d/\d{2,4})", line, flags=re.IGNORECASE)
        if match:
            parsed = parse_date_maybe(match.group(1))
            if parsed:
                return parsed
        match = re.search(r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b", line)
        if match:
            parsed = parse_date_maybe(match.group(0))
            if parsed:
                return parsed
    return None


def extract_neutral_citation(text: str, title: str, existing: Optional[str] = None, url: str = "") -> str:
    citation_re = r"\[\d{4}\]\s*EWHC\s*\d+\s*\((?:KB|QB)\)"
    for source in [existing or "", title or "", text or ""]:
        match = re.search(citation_re, source, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(0)).upper().replace("EWHC ", "EWHC ")

    year = case_year_from_url(url)
    division = case_division_from_url(url)
    num_match = re.search(r"/(\d+)\.html$", url or "")
    if year and division and num_match:
        return f"[{year}] EWHC {num_match.group(1)} ({division})"
    return ""


def extract_case_no(text: str, existing: Optional[str] = None) -> str:
    if existing:
        return str(existing)
    match = re.search(r"\bCase\s*No(?:s)?\s*:?\s*([A-Z0-9][A-Z0-9 ./,&-]{2,80})", text or "", flags=re.IGNORECASE)
    if match:
        return clean_text(match.group(1)).split("\n")[0][:80]
    return ""


def clean_judge_display(name: str) -> str:
    out = clean_text(name or "")
    out = re.sub(r"^\s*:+\s*", "", out)
    out = re.sub(r"\s+", " ", out)
    out = out.strip(" ,;:")
    return out


def canonicalize_judge_name(name: str) -> str:
    out = clean_judge_display(name).upper()
    out = re.sub(r"\bSITTING\s+AS\b.*$", "", out)
    out = re.sub(r"\bSITTING\s+IN\b.*$", "", out)
    out = re.sub(r"\bACTING\s+AS\b.*$", "", out)
    out = out.replace("K.C.", "KC").replace("Q.C.", "QC")
    out = re.sub(r"[.,:;()\[\]]", " ", out)
    out = re.sub(r"\bTHE\s+HONOURABLE\b", " ", out)
    out = re.sub(r"\bTHE\s+HON\b", " ", out)
    out = re.sub(r"\bHONOURABLE\b", " ", out)
    out = re.sub(r"\bHON\b", " ", out)
    out = re.sub(r"\bHER\s+HONOUR\s+JUDGE\b", "HHJ", out)
    out = re.sub(r"\bHIS\s+HONOUR\s+JUDGE\b", "HHJ", out)
    out = re.sub(r"\bSENIOR\s+MASTER\b", "MASTER", out)
    out = re.sub(r"\b(DBE|KC|QC|KBE|CBE|OBE)\b", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out.lower()


def judge_id_from_name(name: str) -> str:
    canon = canonicalize_judge_name(name)
    out = re.sub(r"[^a-z0-9]+", "_", canon).strip("_")
    return out or "unknown_judge"


def is_valid_judge_name(name: str) -> bool:
    if not name:
        return False
    clean = clean_judge_display(name)
    if len(clean) < 4 or len(clean) > 160:
        return False
    lower = clean.lower()
    bad = {
        "between",
        "claimant",
        "defendant",
        "applicant",
        "respondent",
        "hearing date",
        "hearing dates",
        "approved judgment",
        "judgment",
    }
    if any(lower == token or lower.startswith(token + ":") for token in bad):
        return False
    return bool(re.search(r"[A-Za-z]", clean))


def dedupe_judges(judges: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for judge in judges:
        display = clean_judge_display(judge)
        if not is_valid_judge_name(display):
            continue
        jid = judge_id_from_name(display)
        if jid in seen:
            continue
        seen.add(jid)
        out.append(display)
    return out


def extract_judges(lines: List[str], existing: Optional[Sequence[str]] = None) -> List[str]:
    candidates: List[str] = []
    if existing:
        candidates.extend(str(j) for j in existing if j)

    stop_tokens = {
        "between",
        "claimant",
        "claimants",
        "defendant",
        "defendants",
        "applicant",
        "applicants",
        "respondent",
        "respondents",
        "appellant",
        "appellants",
        "hearing date",
        "hearing dates",
        "approved judgment",
    }

    for idx, line in enumerate(lines[:300]):
        if re.match(r"^\s*before\s*:?\s*$", line, flags=re.IGNORECASE) or re.match(r"^\s*before\b", line, flags=re.IGNORECASE):
            same_line = re.sub(r"^\s*before\s*:?\s*", "", line, flags=re.IGNORECASE).strip()
            if same_line:
                candidates.append(same_line)
            for next_line in lines[idx + 1: idx + 7]:
                low = next_line.lower().strip(" :")
                if any(token in low for token in stop_tokens):
                    break
                if len(next_line) <= 160:
                    candidates.append(next_line)
            break

    judge_patterns = [
        r"\b(?:THE\s+HON(?:OURABLE)?\.?\s+)?(?:MR|MRS|MS|MISS)\.?\s+JUSTICE\s+[A-Z][A-Z' -]+(?:\s+(?:DBE|KC|QC|KBE|CBE|OBE))?\b",
        r"\b(?:LORD|LADY)\s+JUSTICE\s+[A-Z][A-Z' -]+\b",
        r"\b(?:HIS|HER)\s+HONOUR\s+JUDGE\s+[A-Z][A-Z' -]+(?:\s+(?:KC|QC))?\b",
        r"\bHHJ\s+[A-Z][A-Z' -]+(?:\s+(?:KC|QC))?\b",
        r"\b(?:SENIOR\s+)?MASTER\s+[A-Z][A-Z' -]+(?:\s+(?:KC|QC))?\b",
        r"\bDEPUTY\s+HIGH\s+COURT\s+JUDGE\s+[A-Z][A-Z' -]+(?:\s+(?:KC|QC))?\b",
    ]
    for line in lines[:180]:
        if len(line) > 180:
            continue
        upper = line.upper()
        for pattern in judge_patterns:
            for match in re.finditer(pattern, upper):
                candidates.append(match.group(0))

    return dedupe_judges(candidates)


TOPIC_PATTERNS: List[Tuple[str, List[str]]] = [
    ("clinical negligence", [r"\bclinical negligence\b", r"\bmedical negligence\b", r"\bnhs trust\b", r"\bhospital\b", r"\bpatient\b"]),
    ("personal injury", [r"\bpersonal injury\b", r"\broad traffic\b", r"\baccident\b", r"\binjury\b", r"\bfatal accident\b"]),
    ("defamation/media/privacy", [r"\bdefamation\b", r"\blibel\b", r"\bprivacy\b", r"\bdata protection\b", r"\bpublication\b", r"\bnewspaper\b", r"\bmedia\b"]),
    ("costs/procedure", [r"\bcosts\b", r"\bsummary judgment\b", r"\bstrike out\b", r"\bpermission to appeal\b", r"\bservice of proceedings\b", r"\bcivil procedure\b"]),
    ("police/prison", [r"\bpolice\b", r"\bprison\b", r"\bparole\b", r"\bcustody\b", r"\bdetention\b"]),
    ("commercial/contract", [r"\bcontract\b", r"\bcommercial\b", r"\bfraud\b", r"\bmisrepresentation\b", r"\binsurance\b", r"\bcompany\b", r"\bdebt\b"]),
    ("contempt", [r"\bcontempt\b", r"\bcommittal\b", r"\binjunction\b"]),
    ("employment", [r"\bemployment\b", r"\bemployee\b", r"\bemployer\b", r"\bworker\b", r"\btrade union\b"]),
    ("housing", [r"\bhousing\b", r"\blandlord\b", r"\btenant\b", r"\bpossession\b", r"\bhomeless\b"]),
    ("immigration/asylum", [r"\bimmigration\b", r"\basylum\b", r"\bhome office\b", r"\bdeportation\b", r"\brefugee\b"]),
    ("inquest/coroner", [r"\binquest\b", r"\bcoroner\b", r"\bprevention of future deaths\b"]),
]


def extract_topic_labels(title: str, text: str) -> List[str]:
    haystack = clean_text(f"{title or ''}\n{(text or '')[:25000]}").lower()
    labels: List[str] = []
    for label, patterns in TOPIC_PATTERNS:
        if any(re.search(pattern, haystack, flags=re.IGNORECASE) for pattern in patterns):
            labels.append(label)
    return labels or ["other"]


def infer_court_division(url: str, neutral_citation: str) -> str:
    from_url = case_division_from_url(url)
    if from_url:
        return from_url
    match = re.search(r"\((KB|QB)\)", neutral_citation or "", flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def parse_case_fields(
    *,
    case_url: str,
    title: str,
    text: str,
    existing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    existing = existing or {}
    lines = text_lines(text)

    neutral_citation = extract_neutral_citation(text, title, existing.get("neutral_citation") or existing.get("case_id"), case_url)
    judgment_date = extract_judgment_date(lines, title, existing.get("judgment_date"))
    hearing_date_raw = extract_hearing_date_raw(lines) or existing.get("hearing_date_raw") or ""
    hearing_start, hearing_end, hearing_warnings = parse_hearing_dates(hearing_date_raw, judgment_date)
    judges = extract_judges(lines, existing.get("justices") or existing.get("judges"))
    judge_display = judges[0] if judges else clean_judge_display(existing.get("judge_display") or existing.get("judge") or "")
    judge_id = judge_id_from_name(judge_display) if judge_display else ""
    court_division = infer_court_division(case_url, neutral_citation)
    case_no = extract_case_no(text, existing.get("case_no"))
    topic_labels = extract_topic_labels(title, text)

    reserved_days = days_between(hearing_end, judgment_date)
    hearing_span_days = None
    if hearing_start and hearing_end:
        span = days_between(hearing_start, hearing_end)
        hearing_span_days = span + 1 if span is not None and span >= 0 else None

    warnings = list(hearing_warnings)
    if reserved_days is not None and reserved_days > 365:
        warnings.append("reserved_days_over_one_year")
    if reserved_days is not None and reserved_days < 0:
        warnings.append("negative_reserved_days")
    if not judge_display:
        warnings.append("missing_judge")
    if not judgment_date:
        warnings.append("missing_judgment_date")
    if not hearing_end:
        warnings.append("missing_hearing_end")
    if not neutral_citation:
        warnings.append("missing_neutral_citation")
    if not court_division:
        warnings.append("missing_court_division")

    word_count = len(re.findall(r"\b\w+\b", text or ""))
    paragraph_count = max(1, len([ln for ln in (text or "").splitlines() if ln.strip()]))

    return {
        **existing,
        "case_url": case_url,
        "judgment_url": case_url,
        "title": title or existing.get("title", ""),
        "case_id": neutral_citation,
        "neutral_citation": neutral_citation,
        "case_no": case_no,
        "court_division": court_division,
        "judgment_date": judgment_date,
        "hearing_date_raw": hearing_date_raw,
        "hearing_start": hearing_start,
        "hearing_end": hearing_end,
        "hearing_span_days": hearing_span_days,
        "reserved_days": reserved_days,
        "justices": judges[:3] if judges else ([judge_display] if judge_display else []),
        "judge_display": judge_display,
        "judge_id": judge_id,
        "topic_labels": topic_labels,
        "parse_warnings": sorted(set(warnings)),
        "text_char_count": len(text or ""),
        "word_count": word_count,
        "paragraph_count": paragraph_count,
        "analysis_text": text or existing.get("analysis_text", ""),
        "text_full": existing.get("text_full") or text,
        "_parse_version": CASE_PARSE_VERSION,
    }


def parse_bailii_case(case_url: str, session: requests.Session) -> Dict[str, Any]:
    response = request_get(case_url, session)
    if response is None or response.status_code != 200:
        raise RuntimeError(f"Failed to fetch {case_url}")
    title, text, _ = html_to_text_and_links(response.text)
    if not title:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", response.text, flags=re.IGNORECASE | re.DOTALL)
        title = clean_text(re.sub(r"<[^>]+>", " ", title_match.group(1))) if title_match else ""
    return parse_case_fields(case_url=case_url, title=title, text=text, existing={})


def normalize_cached_case(case: Dict[str, Any]) -> Dict[str, Any]:
    case_url = case.get("case_url") or case.get("judgment_url") or ""
    title = case.get("title") or ""
    text = case.get("analysis_text") or case.get("text_full") or ""
    return parse_case_fields(case_url=case_url, title=title, text=text, existing=case)


def load_cases(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return data["cases"]
    raise ValueError(f"Unexpected cases cache format at {path}")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def parse_max_cases(value: str) -> Optional[int]:
    if str(value).lower() == "all":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max-cases must be a positive integer or 'all'")
    return parsed


def crawl_cases(
    *,
    session: requests.Session,
    max_cases: Optional[int],
    seed_cases: List[Dict[str, Any]],
    refresh_parse: bool,
    court_scope: str,
) -> List[Dict[str, Any]]:
    divisions = COURT_DIVISIONS if court_scope == "kb-qb" else ("KB",)
    normalized: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    if seed_cases and not refresh_parse:
        print(f"[cache] normalizing {len(seed_cases)} cached cases from {CASES_RAW_PATH}")
        for case in tqdm(seed_cases, desc="Normalizing cached cases"):
            normalized_case = normalize_cached_case(case)
            url = normalized_case.get("case_url")
            if not url or url in seen_urls:
                continue
            if normalized_case.get("court_division") in divisions:
                normalized.append(normalized_case)
                seen_urls.add(url)
            if max_cases is not None and len(normalized) >= max_cases:
                return normalized[:max_cases]

    if max_cases is not None and len(normalized) >= max_cases:
        return normalized[:max_cases]

    print("[crawl] discovering BAILII KB/QB case URLs")
    urls = list_bailii_case_urls(session, divisions)
    if max_cases is not None:
        urls = urls[:max_cases]
    print(f"[crawl] discovered {len(urls)} candidate URLs")

    last_saved = len(normalized)
    for url in tqdm(urls, desc="Crawling BAILII cases"):
        if url in seen_urls:
            continue
        try:
            case = parse_bailii_case(url, session)
        except BailiiBlockedError:
            raise
        except Exception as exc:
            case = {
                "case_url": url,
                "parse_warnings": [f"fetch_or_parse_error:{type(exc).__name__}"],
                "_parse_version": CASE_PARSE_VERSION,
            }
        normalized.append(case)
        seen_urls.add(url)
        time.sleep(REQUEST_DELAY_SEC)

        if len(normalized) - last_saved >= CACHE_SAVE_EVERY:
            save_json(CASES_RAW_PATH, normalized)
            last_saved = len(normalized)

        if max_cases is not None and len(normalized) >= max_cases:
            break

    return normalized[:max_cases] if max_cases is not None else normalized


# -----------------------------
# Model-ready dataset
# -----------------------------


def year_from_iso(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        return int(value[:4])
    except Exception:
        return None


def month_from_iso(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        return int(value[5:7])
    except Exception:
        return None


def quarter_from_month(month: Optional[int]) -> Optional[int]:
    if not month:
        return None
    return int((month - 1) // 3 + 1)


def validate_case(case: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    required = [
        ("case_url", case.get("case_url")),
        ("judge_id", case.get("judge_id")),
        ("judge_display", case.get("judge_display")),
        ("judgment_date", case.get("judgment_date")),
        ("hearing_end", case.get("hearing_end")),
        ("reserved_days", case.get("reserved_days")),
        ("court_division", case.get("court_division")),
        ("neutral_citation", case.get("neutral_citation") or case.get("case_id")),
    ]
    for name, value in required:
        if value in (None, "", []):
            reasons.append(f"missing_{name}")
    if case.get("reserved_days") is not None and safe_float(case.get("reserved_days"), -1) < 0:
        reasons.append("negative_reserved_days")
    if case.get("court_division") not in COURT_DIVISIONS:
        reasons.append("unsupported_court_division")
    return not reasons, reasons


def build_speed_dataset(cases: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rejections: List[Dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()

    for case in cases:
        for warning in case.get("parse_warnings", []) or []:
            warning_counts[str(warning)] += 1
        ok, reasons = validate_case(case)
        if not ok:
            rejections.append({
                "case_url": case.get("case_url"),
                "case_id": case.get("case_id") or case.get("neutral_citation"),
                "title": case.get("title", ""),
                "reasons": sorted(set(reasons)),
                "parse_warnings": case.get("parse_warnings", []),
            })
            continue

        judgment_year = year_from_iso(case.get("judgment_date"))
        judgment_month = month_from_iso(case.get("judgment_date"))
        topics = case.get("topic_labels") or ["other"]
        row = {
            "case_id": case.get("neutral_citation") or case.get("case_id"),
            "neutral_citation": case.get("neutral_citation") or case.get("case_id"),
            "title": case.get("title", ""),
            "case_url": case.get("case_url"),
            "case_no": case.get("case_no", ""),
            "court_division": case.get("court_division"),
            "judge": case.get("judge_display"),
            "judge_display": case.get("judge_display"),
            "judge_id": case.get("judge_id"),
            "hearing_date_raw": case.get("hearing_date_raw", ""),
            "hearing_start": case.get("hearing_start"),
            "hearing_end": case.get("hearing_end"),
            "judgment_date": case.get("judgment_date"),
            "judgment_year": judgment_year,
            "judgment_month": judgment_month,
            "judgment_quarter": quarter_from_month(judgment_month),
            "year_month": f"{judgment_year}-{judgment_month:02d}" if judgment_year and judgment_month else "",
            "reserved_days": int(case.get("reserved_days")),
            "hearing_span_days": int(case.get("hearing_span_days") or 1),
            "topic_labels": topics,
            "primary_topic": topics[0] if topics else "other",
            "text_char_count": int(case.get("text_char_count") or len(case.get("analysis_text", ""))),
            "word_count": int(case.get("word_count") or 0),
            "paragraph_count": int(case.get("paragraph_count") or 1),
            "parse_warnings": case.get("parse_warnings", []),
        }
        rows.append(row)

    parse_quality = {
        "raw_cases": len(cases),
        "model_ready_cases": len(rows),
        "rejected_cases": len(rejections),
        "warning_counts": dict(warning_counts.most_common()),
        "topic_counts": dict(Counter(label for row in rows for label in row.get("topic_labels", [])).most_common()),
        "court_counts": dict(Counter(row["court_division"] for row in rows).most_common()),
    }
    return rows, rejections, parse_quality


def build_judge_aliases(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    aliases: Dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        judge_id = case.get("judge_id")
        display = case.get("judge_display")
        if judge_id and display:
            aliases[judge_id][display] += 1
        for judge in case.get("justices", []) or []:
            if judge:
                aliases[judge_id_from_name(judge)][clean_judge_display(judge)] += 1
    return {
        judge_id: {
            "judge_id": judge_id,
            "display": counts.most_common(1)[0][0],
            "aliases": [{"name": name, "count": count} for name, count in counts.most_common()],
        }
        for judge_id, counts in sorted(aliases.items())
        if judge_id
    }


# -----------------------------
# Regularized model
# -----------------------------


@dataclass
class FeaturePlan:
    min_judge_cases: int
    eligible_judges: List[str]
    category_values: Dict[str, List[str]]
    numeric_means: Dict[str, float]
    numeric_stds: Dict[str, float]
    col_names: List[str]


NUMERIC_FEATURES = ["hearing_span_days", "text_char_count", "word_count", "paragraph_count", "topic_count"]
CATEGORY_FEATURES = ["year_month", "court_division", "primary_topic"]


def build_feature_plan(rows: List[Dict[str, Any]], min_judge_cases: int) -> FeaturePlan:
    judge_counts = Counter(row["judge_id"] for row in rows)
    # Fit a regularized effect for every observed judge. The minimum-case
    # threshold is applied later as an assignment/scenario eligibility filter,
    # so the dashboard can explore lower or higher thresholds without refitting.
    eligible_judges = sorted(judge_counts)

    category_values: Dict[str, List[str]] = {}
    for feature in CATEGORY_FEATURES:
        category_values[feature] = sorted({str(row.get(feature) or "") for row in rows if row.get(feature) not in (None, "")})

    numeric_values: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        row["topic_count"] = len(row.get("topic_labels") or [])
        for feature in NUMERIC_FEATURES:
            numeric_values[feature].append(math.log1p(max(0.0, safe_float(row.get(feature)))))

    numeric_means: Dict[str, float] = {}
    numeric_stds: Dict[str, float] = {}
    for feature in NUMERIC_FEATURES:
        arr = np.asarray(numeric_values[feature], dtype=float)
        numeric_means[feature] = float(arr.mean()) if arr.size else 0.0
        std = float(arr.std()) if arr.size else 1.0
        numeric_stds[feature] = std if std > 1e-9 else 1.0

    col_names = ["intercept"]
    col_names.extend(f"judge:{judge}" for judge in eligible_judges)
    for feature, values in category_values.items():
        col_names.extend(f"{feature}:{value}" for value in values)
    col_names.extend(f"num:{feature}" for feature in NUMERIC_FEATURES)

    return FeaturePlan(
        min_judge_cases=min_judge_cases,
        eligible_judges=eligible_judges,
        category_values=category_values,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
        col_names=col_names,
    )


def row_to_features(row: Dict[str, Any], plan: FeaturePlan, override_judge_id: Optional[str] = None) -> np.ndarray:
    values: List[float] = [1.0]
    judge_id = override_judge_id or row.get("judge_id")
    values.extend(1.0 if judge_id == eligible else 0.0 for eligible in plan.eligible_judges)
    for feature in CATEGORY_FEATURES:
        row_value = str(row.get(feature) or "")
        values.extend(1.0 if row_value == category else 0.0 for category in plan.category_values[feature])
    for feature in NUMERIC_FEATURES:
        raw = math.log1p(max(0.0, safe_float(row.get(feature))))
        values.append((raw - plan.numeric_means[feature]) / plan.numeric_stds[feature])
    return np.asarray(values, dtype=float)


def build_matrix(rows: List[Dict[str, Any]], plan: FeaturePlan) -> np.ndarray:
    return np.vstack([row_to_features(row, plan) for row in rows])


def fit_ridge(rows: List[Dict[str, Any]], target_key: str, plan: FeaturePlan, alpha: float = RIDGE_ALPHA) -> Dict[str, Any]:
    X = build_matrix(rows, plan)
    y = np.asarray([safe_float(row[target_key]) for row in rows], dtype=float)
    penalty = np.ones(X.shape[1], dtype=float) * alpha
    penalty[0] = 0.0
    beta = np.linalg.solve(X.T @ X + np.diag(penalty), X.T @ y)
    pred = X @ beta
    residual = y - pred
    ssr = float(residual @ residual)
    sst = float(((y - y.mean()) @ (y - y.mean()))) if y.size else 0.0
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    judge_effects = {
        name.replace("judge:", "", 1): float(coef)
        for name, coef in zip(plan.col_names, beta)
        if name.startswith("judge:")
    }
    return {
        "target_key": target_key,
        "beta": beta,
        "pred": pred,
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(residual ** 2))) if residual.size else 0.0,
        "judge_effects": judge_effects,
        "col_names": plan.col_names,
    }


def _continued_fraction_beta(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-14) -> float:
    """Continued fraction used by the regularized incomplete beta function."""
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not (a > 0 and b > 0) or not math.isfinite(x):
        return float("nan")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        value = bt * _continued_fraction_beta(a, b, x) / a
    else:
        value = 1.0 - bt * _continued_fraction_beta(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, float(value)))


def f_survival_p_value(f_stat: float, df_num: int, df_den: int) -> Optional[float]:
    if not (math.isfinite(f_stat) and f_stat >= 0 and df_num > 0 and df_den > 0):
        return None
    x = (df_num * f_stat) / (df_num * f_stat + df_den)
    cdf = regularized_incomplete_beta(df_num / 2.0, df_den / 2.0, x)
    if not math.isfinite(cdf):
        return None
    return min(1.0, max(0.0, 1.0 - cdf))


def build_fixed_effect_matrix(rows: List[Dict[str, Any]], plan: FeaturePlan, include_judges: bool) -> Tuple[np.ndarray, List[str]]:
    col_names = ["intercept"]
    if include_judges:
        col_names.extend(f"judge:{judge}" for judge in plan.eligible_judges[1:])
    for feature, values in plan.category_values.items():
        col_names.extend(f"{feature}:{value}" for value in values[1:])
    col_names.extend(f"num:{feature}" for feature in NUMERIC_FEATURES)

    matrix: List[List[float]] = []
    for row in rows:
        values: List[float] = [1.0]
        if include_judges:
            judge_id = row.get("judge_id")
            values.extend(1.0 if judge_id == judge else 0.0 for judge in plan.eligible_judges[1:])
        for feature in CATEGORY_FEATURES:
            row_value = str(row.get(feature) or "")
            values.extend(1.0 if row_value == category else 0.0 for category in plan.category_values[feature][1:])
        for feature in NUMERIC_FEATURES:
            raw = math.log1p(max(0.0, safe_float(row.get(feature))))
            values.append((raw - plan.numeric_means[feature]) / plan.numeric_stds[feature])
        matrix.append(values)
    return np.asarray(matrix, dtype=float), col_names


def fit_ols_matrix(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    residual = y - pred
    rss = float(residual @ residual)
    sst = float(((y - y.mean()) @ (y - y.mean()))) if y.size else 0.0
    r2 = 1.0 - rss / sst if sst > 0 else 0.0
    df_resid = int(max(0, y.size - rank))
    adj_r2 = 1.0 - (rss / df_resid) / (sst / (y.size - 1)) if sst > 0 and df_resid > 0 and y.size > 1 else r2
    return {
        "beta": beta,
        "pred": pred,
        "rss": rss,
        "rank": int(rank),
        "df_resid": df_resid,
        "r2": float(r2),
        "adj_r2": float(adj_r2),
    }


def dispersion_summary(values: Sequence[float]) -> Dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"std": 0.0, "p10": 0.0, "p90": 0.0, "iqr": 0.0, "range": 0.0, "min": 0.0, "max": 0.0}
    return {
        "std": float(np.std(finite)),
        "p10": percentile(finite, 10),
        "p90": percentile(finite, 90),
        "iqr": float(percentile(finite, 75) - percentile(finite, 25)),
        "range": float(max(finite) - min(finite)),
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def fixed_effect_diagnostic_for_target(rows: List[Dict[str, Any]], plan: FeaturePlan, target_key: str, target_label: str) -> Dict[str, Any]:
    y = np.asarray([safe_float(row[target_key]) for row in rows], dtype=float)
    restricted_x, _ = build_fixed_effect_matrix(rows, plan, include_judges=False)
    unrestricted_x, unrestricted_cols = build_fixed_effect_matrix(rows, plan, include_judges=True)
    restricted = fit_ols_matrix(restricted_x, y)
    unrestricted = fit_ols_matrix(unrestricted_x, y)

    df_num = int(max(0, unrestricted["rank"] - restricted["rank"]))
    df_den = int(unrestricted["df_resid"])
    if df_num > 0 and df_den > 0 and unrestricted["rss"] > 1e-12:
        f_stat = max(0.0, ((restricted["rss"] - unrestricted["rss"]) / df_num) / (unrestricted["rss"] / df_den))
        p_value = f_survival_p_value(f_stat, df_num, df_den)
    else:
        f_stat = 0.0
        p_value = None

    baseline_judge = plan.eligible_judges[0] if plan.eligible_judges else ""
    judge_effects = {baseline_judge: 0.0} if baseline_judge else {}
    for name, coef in zip(unrestricted_cols, unrestricted["beta"]):
        if name.startswith("judge:"):
            judge_effects[name.replace("judge:", "", 1)] = float(coef)
    if judge_effects:
        mean_effect = float(np.mean(list(judge_effects.values())))
        judge_effects = {judge: float(effect - mean_effect) for judge, effect in judge_effects.items()}

    return {
        "target": target_label,
        "n_cases": len(rows),
        "n_judges": len(plan.eligible_judges),
        "f_test": {
            "stat": float(f_stat),
            "p_value": p_value,
            "df_num": df_num,
            "df_den": df_den,
            "rss_restricted": float(restricted["rss"]),
            "rss_unrestricted": float(unrestricted["rss"]),
        },
        "restricted_r2": float(restricted["r2"]),
        "unrestricted_r2": float(unrestricted["r2"]),
        "unrestricted_adj_r2": float(unrestricted["adj_r2"]),
        "dispersion": dispersion_summary(list(judge_effects.values())),
    }


def run_fixed_effect_diagnostics(rows: List[Dict[str, Any]], plan: FeaturePlan) -> Dict[str, Any]:
    return {
        "log": fixed_effect_diagnostic_for_target(rows, plan, "reserved_log", "log1p(reserved_days)"),
        "raw": fixed_effect_diagnostic_for_target(rows, plan, "reserved_days", "reserved_days"),
    }


def add_model_predictions(rows: List[Dict[str, Any]], plan: FeaturePlan, log_model: Dict[str, Any], raw_model: Dict[str, Any]) -> None:
    log_effects = log_model["judge_effects"]
    raw_effects = raw_model["judge_effects"]
    log_pred = log_model["pred"]
    raw_pred = raw_model["pred"]

    for idx, row in enumerate(rows):
        judge_id = row["judge_id"]
        log_eff = float(log_effects.get(judge_id, 0.0))
        raw_eff = float(raw_effects.get(judge_id, 0.0))
        row["reserved_log"] = math.log1p(row["reserved_days"])
        row["pred_log"] = float(log_pred[idx])
        row["pred_raw"] = max(0.0, float(raw_pred[idx]))
        row["pred_days_log_model"] = max(0.0, math.expm1(float(log_pred[idx])))
        row["base_pred_log"] = float(log_pred[idx]) - log_eff
        row["base_pred_raw"] = max(0.0, float(raw_pred[idx]) - raw_eff)
        row["judge_effect_log"] = log_eff
        row["judge_effect_raw"] = raw_eff


def summarize_turnaround(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = [float(row["reserved_days"]) for row in rows]
    if not values:
        return {}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": percentile(values, 10),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def build_judge_effect_rows(rows: List[Dict[str, Any]], aliases: Dict[str, Any], log_model: Dict[str, Any], raw_model: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_judge: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_judge[row["judge_id"]].append(row)

    out: List[Dict[str, Any]] = []
    for judge_id, judge_rows in by_judge.items():
        reserved = [row["reserved_days"] for row in judge_rows]
        topics = Counter(label for row in judge_rows for label in row.get("topic_labels", []))
        display = aliases.get(judge_id, {}).get("display") or judge_rows[0]["judge_display"]
        effect_log = float(log_model["judge_effects"].get(judge_id, 0.0))
        effect_raw = float(raw_model["judge_effects"].get(judge_id, 0.0))
        out.append({
            "judge_id": judge_id,
            "judge": display,
            "judge_display": display,
            "n_cases": len(judge_rows),
            "mean_reserved_days": float(np.mean(reserved)),
            "median_reserved_days": float(np.median(reserved)),
            "effect_log": effect_log,
            "effect_days_at_mean": float(math.expm1(max(0.0, math.log1p(np.mean(reserved)) + effect_log)) - np.mean(reserved)),
            "effect_raw_days": effect_raw,
            "eligible_for_assignment": judge_id in log_model["judge_effects"],
            "top_topics": [{"topic": topic, "count": count} for topic, count in topics.most_common(5)],
            "aliases": aliases.get(judge_id, {}).get("aliases", []),
        })
    out.sort(key=lambda item: (item["effect_log"], item["judge"]))
    return out


# -----------------------------
# Counterfactual assignment
# -----------------------------


def bucket_for_row(row: Dict[str, Any], time_bucket: str) -> str:
    year = row.get("judgment_year")
    month = row.get("judgment_month")
    if time_bucket == "month" and year and month:
        return f"{year}-{month:02d}"
    if time_bucket == "quarter" and year and month:
        return f"{year}-Q{quarter_from_month(month)}"
    return str(year or "")


def predict_days_for_judge(row: Dict[str, Any], judge_id: str, model_view: str, judge_effects: Dict[str, float]) -> float:
    effect = float(judge_effects.get(judge_id, 0.0))
    if model_view == "raw":
        return max(0.0, float(row.get("base_pred_raw", 0.0)) + effect)
    pred_log = float(row.get("base_pred_log", 0.0)) + effect
    return max(0.0, math.expm1(pred_log))


def run_assignment(
    rows: List[Dict[str, Any]],
    judge_effects: Dict[str, float],
    *,
    min_judge_cases: int,
    capacity_slack: float,
    time_bucket: str,
    model_view: str = "log",
) -> Dict[str, Any]:
    judge_counts_all = Counter(row["judge_id"] for row in rows)
    eligible_global = {judge for judge, count in judge_counts_all.items() if count >= min_judge_cases}
    assignments: List[Dict[str, Any]] = []

    for bucket, bucket_rows in group_rows(rows, lambda row: bucket_for_row(row, time_bucket)).items():
        bucket_counts = Counter(row["judge_id"] for row in bucket_rows)
        eligible = sorted(judge for judge in bucket_counts if judge in eligible_global and judge in judge_effects)
        extra_capacity = {
            judge: max(0, int(math.ceil(bucket_counts[judge] * (1.0 + capacity_slack))) - bucket_counts[judge])
            for judge in eligible
        }

        planned: List[Tuple[float, Dict[str, Any], Optional[str], float]] = []
        for row in bucket_rows:
            current_pred = predict_days_for_judge(row, row["judge_id"], model_view, judge_effects)
            best_judge = None
            best_pred = current_pred
            for candidate in eligible:
                if candidate == row["judge_id"] or extra_capacity.get(candidate, 0) <= 0:
                    continue
                candidate_pred = predict_days_for_judge(row, candidate, model_view, judge_effects)
                if candidate_pred < best_pred:
                    best_pred = candidate_pred
                    best_judge = candidate
            planned.append((current_pred - best_pred, row, best_judge, best_pred))

        planned.sort(key=lambda item: item[0], reverse=True)
        for savings, row, best_judge, best_pred in planned:
            current_pred = predict_days_for_judge(row, row["judge_id"], model_view, judge_effects)
            to_judge = row["judge_id"]
            reassigned_pred = current_pred
            moved = False
            if best_judge and savings > 1e-6 and extra_capacity.get(best_judge, 0) > 0:
                to_judge = best_judge
                reassigned_pred = best_pred
                extra_capacity[best_judge] -= 1
                moved = True
            assignments.append({
                "case_id": row["case_id"],
                "title": row["title"],
                "case_url": row["case_url"],
                "bucket": bucket,
                "judgment_year": row["judgment_year"],
                "court_division": row["court_division"],
                "topic_labels": row["topic_labels"],
                "from_judge_id": row["judge_id"],
                "from_judge": row["judge_display"],
                "to_judge_id": to_judge,
                "to_judge": to_judge,
                "moved": moved,
                "actual_reserved_days": row["reserved_days"],
                "current_pred_days": float(current_pred),
                "reassigned_pred_days": float(reassigned_pred),
                "predicted_savings_days": float(max(0.0, current_pred - reassigned_pred)),
            })

    display_by_id = {row["judge_id"]: row["judge_display"] for row in rows}
    for assignment in assignments:
        assignment["to_judge"] = display_by_id.get(assignment["to_judge_id"], assignment["to_judge_id"])

    return summarize_assignments(assignments, rows, min_judge_cases, capacity_slack, time_bucket, model_view)


def group_rows(rows: List[Dict[str, Any]], key_fn: Any) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(row)
    return grouped


def summarize_assignments(
    assignments: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    min_judge_cases: int,
    capacity_slack: float,
    time_bucket: str,
    model_view: str,
) -> Dict[str, Any]:
    current_total = sum(item["current_pred_days"] for item in assignments)
    reassigned_total = sum(item["reassigned_pred_days"] for item in assignments)
    moved = [item for item in assignments if item["moved"]]
    action_counts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in moved:
        key = (item["from_judge_id"], item["to_judge_id"])
        action = action_counts.setdefault(key, {
            "from_judge_id": item["from_judge_id"],
            "from_judge": item["from_judge"],
            "to_judge_id": item["to_judge_id"],
            "to_judge": item["to_judge"],
            "cases": 0,
            "predicted_savings_days": 0.0,
        })
        action["cases"] += 1
        action["predicted_savings_days"] += item["predicted_savings_days"]

    top_actions = list(action_counts.values())
    for action in top_actions:
        action["avg_savings_days"] = action["predicted_savings_days"] / max(1, action["cases"])
    top_actions.sort(key=lambda item: item["predicted_savings_days"], reverse=True)

    current_counts = Counter(row["judge_id"] for row in rows)
    reassigned_counts = Counter(item["to_judge_id"] for item in assignments)
    inflow = Counter(item["to_judge_id"] for item in moved)
    outflow = Counter(item["from_judge_id"] for item in moved)
    display_by_id = {row["judge_id"]: row["judge_display"] for row in rows}
    judge_flow = []
    for judge_id in sorted(set(current_counts) | set(reassigned_counts)):
        judge_flow.append({
            "judge_id": judge_id,
            "judge": display_by_id.get(judge_id, judge_id),
            "current_cases": current_counts.get(judge_id, 0),
            "reassigned_cases": reassigned_counts.get(judge_id, 0),
            "inflow": inflow.get(judge_id, 0),
            "outflow": outflow.get(judge_id, 0),
            "net": reassigned_counts.get(judge_id, 0) - current_counts.get(judge_id, 0),
        })
    judge_flow.sort(key=lambda item: (abs(item["net"]), item["inflow"] + item["outflow"]), reverse=True)

    return {
        "settings": {
            "min_judge_cases": min_judge_cases,
            "capacity_slack": capacity_slack,
            "time_bucket": time_bucket,
            "model_view": model_view,
        },
        "summary": {
            "n_cases": len(assignments),
            "moved_cases": len(moved),
            "current_total_pred_days": float(current_total),
            "reassigned_total_pred_days": float(reassigned_total),
            "predicted_savings_days": float(current_total - reassigned_total),
            "predicted_savings_pct": float((current_total - reassigned_total) / current_total * 100.0) if current_total > 0 else 0.0,
            "avg_current_pred_days": float(current_total / len(assignments)) if assignments else 0.0,
            "avg_reassigned_pred_days": float(reassigned_total / len(assignments)) if assignments else 0.0,
            "avg_actual_reserved_days": float(np.mean([row["reserved_days"] for row in rows])) if rows else 0.0,
        },
        "assignments": assignments,
        "top_actions": top_actions[:100],
        "judge_flow": judge_flow,
    }


# -----------------------------
# Dashboard
# -----------------------------


def dashboard_float(value: Any, digits: int = 6) -> float:
    return round(safe_float(value), digits)


def build_dashboard_data(
    rows: List[Dict[str, Any]],
    judge_rows: List[Dict[str, Any]],
    assignment_results: Dict[str, Any],
    parse_quality: Dict[str, Any],
    log_model: Dict[str, Any],
    raw_model: Dict[str, Any],
    fixed_effect_diagnostics: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    years = sorted({row["judgment_year"] for row in rows if row.get("judgment_year")})
    topics = sorted({label for row in rows for label in row.get("topic_labels", [])})
    dashboard_cases = []
    for row in rows:
        dashboard_cases.append({
            "case_id": row["case_id"],
            "title": row["title"],
            "case_url": row["case_url"],
            "court_division": row["court_division"],
            "judge_id": row["judge_id"],
            "judge": row["judge_display"],
            "judgment_year": row["judgment_year"],
            "judgment_month": row["judgment_month"],
            "judgment_quarter": row["judgment_quarter"],
            "year_month": row["year_month"],
            "topic_labels": row["topic_labels"],
            "base_pred_log": dashboard_float(row["base_pred_log"]),
            "base_pred_raw": dashboard_float(row["base_pred_raw"]),
        })

    dashboard_judges = []
    for row in judge_rows:
        dashboard_judges.append({
            "judge_id": row["judge_id"],
            "judge": row.get("judge") or row.get("judge_display") or row["judge_id"],
            "judge_display": row.get("judge_display") or row.get("judge") or row["judge_id"],
            "n_cases": row["n_cases"],
            "effect_log": dashboard_float(row["effect_log"]),
            "effect_raw_days": dashboard_float(row["effect_raw_days"]),
            "effect_days_at_mean": dashboard_float(row["effect_days_at_mean"]),
            "eligible_for_assignment": row["eligible_for_assignment"],
        })

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "min_judge_cases": args.min_judge_cases,
            "capacity_slack": args.capacity_slack,
            "time_bucket": args.time_bucket,
            "metric": args.metric,
            "court_scope": args.court_scope,
        },
        "years": years,
        "courts": sorted({row["court_division"] for row in rows}),
        "topics": topics,
        "cases": dashboard_cases,
        "judges": dashboard_judges,
        "parse_quality": parse_quality,
        "model_stats": {
            "log": {
                "r2": log_model["r2"],
                "rmse": log_model["rmse"],
                "target": "log1p(reserved_days)",
            },
            "raw": {
                "r2": raw_model["r2"],
                "rmse": raw_model["rmse"],
                "target": "reserved_days",
            },
            "fixed_effect_diagnostics": fixed_effect_diagnostics,
        },
        "assignment_results": {
            "summary": assignment_results["summary"],
            "top_actions": assignment_results["top_actions"][:25],
            "judge_flow": assignment_results["judge_flow"][:50],
        },
    }


def compact_json_for_html(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def write_dashboard_html(path: Path, dashboard_data: Dict[str, Any]) -> None:
    embedded_data = compact_json_for_html(dashboard_data)
    html_doc = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Judge Reassignment Analysis</title>
  <style>
    :root {
      --ink: #20242a;
      --muted: #667085;
      --line: #d9dee7;
      --brand: #c83b18;
      --brand-dark: #94250e;
      --blue: #2266aa;
      --teal: #16817a;
      --gold: #c28410;
      --bg: #f6f7f9;
      --panel: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      background: #0f766e;
      border-bottom: 4px solid #f59e0b;
      color: white;
      padding: 28px 32px 22px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 34px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .subhead {
      color: #e8fffb;
      max-width: 1100px;
      line-height: 1.45;
    }
    .data-note {
      margin-top: 12px;
      color: #d7fbf5;
      font-size: 13px;
      line-height: 1.45;
      max-width: 1100px;
    }
    main {
      padding: 24px 32px 38px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(8, minmax(120px, 1fr));
      gap: 14px;
      padding: 18px 0 22px;
      border-bottom: 1px solid var(--line);
      align-items: end;
    }
    label {
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
      color: var(--muted);
      margin-bottom: 5px;
      font-weight: 700;
    }
    select, input {
      width: 100%;
      height: 36px;
      border: 1px solid #bfc7d4;
      border-radius: 6px;
      background: white;
      color: var(--ink);
      padding: 0 9px;
      font-size: 14px;
    }
    button {
      height: 36px;
      border: 1px solid var(--brand-dark);
      border-radius: 6px;
      background: var(--brand);
      color: white;
      padding: 0 14px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--brand-dark); }
    .kpis {
      display: grid;
      grid-template-columns: repeat(5, minmax(160px, 1fr));
      gap: 12px;
      margin: 20px 0;
    }
    .kpi {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 15px;
      min-height: 88px;
    }
    .kpi .label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    .kpi .value {
      font-size: 28px;
      font-weight: 750;
      margin-top: 8px;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(520px, 1.05fr) minmax(480px, .95fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }
    .panel h2 {
      margin: 0 0 12px;
      font-size: 18px;
    }
    .chart {
      width: 100%;
      min-height: 280px;
    }
    .grid2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-top: 18px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th {
      text-align: left;
      border-bottom: 2px solid #2c3138;
      padding: 8px 7px;
      color: #353b43;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .03em;
      cursor: pointer;
      user-select: none;
    }
    td {
      border-bottom: 1px solid var(--line);
      padding: 8px 7px;
      vertical-align: top;
    }
    tbody tr:nth-child(even) { background: #fafbfc; }
    .number { text-align: right; font-variant-numeric: tabular-nums; }
    .pill {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: #eef3f8;
      color: #405061;
      margin: 1px 3px 1px 0;
      white-space: nowrap;
      font-size: 12px;
    }
    .quality {
      display: grid;
      grid-template-columns: repeat(4, minmax(110px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .quality div {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
    }
    .unit-note {
      margin: 12px 0 0;
      color: #475467;
      font-size: 13px;
      line-height: 1.45;
      max-width: 1180px;
    }
    .control-guide {
      margin-top: 14px;
      padding: 0 0 18px;
      border-bottom: 1px solid var(--line);
      color: #475467;
      font-size: 13px;
      line-height: 1.45;
    }
    .control-guide h2 {
      margin: 0 0 8px;
      font-size: 16px;
      color: var(--ink);
    }
    .guide-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 10px 18px;
      max-width: 1180px;
    }
    .guide-grid strong {
      color: var(--ink);
    }
    .small { color: var(--muted); font-size: 12px; }
    canvas { width: 100%; height: 280px; display: block; }
    svg { width: 100%; height: 290px; display: block; }
    a { color: var(--blue); text-decoration: none; }
    a:hover { text-decoration: underline; }
    @media (max-width: 1120px) {
      .controls { grid-template-columns: repeat(3, minmax(140px, 1fr)); }
      .guide-grid { grid-template-columns: 1fr; }
      .kpis { grid-template-columns: repeat(2, minmax(160px, 1fr)); }
      .layout, .grid2 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Judge Reassignment Analysis</h1>
    <div class="subhead">Counterfactual assignment model for EWHC King's/Queen's Bench cases. Primary speed is the number of calendar days from hearing end to judgment.</div>
    <div class="data-note" id="dataNote">Built from a cached BAILII sample; loading summary...</div>
  </header>
  <main>
    <section class="controls">
      <div><label for="yearStart">Start year</label><select id="yearStart"></select></div>
      <div><label for="yearEnd">End year</label><select id="yearEnd"></select></div>
      <div><label for="court">Court</label><select id="court"><option value="all">All</option></select></div>
      <div><label for="topic">Topic</label><select id="topic"><option value="all">All</option></select></div>
      <div><label for="minCases">Min judge cases</label><input id="minCases" type="number" min="1" step="1" value="5" /></div>
      <div><label for="capacitySlack">Capacity slack</label><input id="capacitySlack" type="number" min="0" max="1" step="0.01" value="0.10" /></div>
      <div><label for="timeBucket">Time bucket</label><select id="timeBucket"><option value="year">Year</option><option value="quarter">Quarter</option><option value="month">Month</option></select></div>
      <div><label for="modelView">Model view</label><select id="modelView"><option value="log">Log model</option><option value="raw">Raw-days model</option></select></div>
      <div><label>&nbsp;</label><button id="applyFilters" type="button">Apply filters</button></div>
    </section>
    <section class="control-guide">
      <h2>How to read the controls</h2>
      <div class="guide-grid">
        <div><strong>Year, court, and topic</strong> choose which cases are included in the scenario.</div>
        <div><strong>Min judge cases</strong> prevents the model from moving cases to judges with very little observed data.</div>
        <div><strong>Capacity slack</strong> controls how much additional work a judge may receive. A value of 0.10 means up to 10% more cases than that judge handled in the selected time bucket.</div>
        <div><strong>Time bucket</strong> keeps assignments realistic by only moving cases among judges active in the same year, quarter, or month.</div>
        <div><strong>Model view</strong> switches between the main log model and a raw-days comparison. Use the log model unless you specifically want day-for-day sensitivity to very long cases.</div>
        <div><strong>Predicted savings</strong> means model-estimated hearing-to-judgment days saved, not guaranteed calendar time saved by the court.</div>
      </div>
    </section>

    <section class="kpis">
      <div class="kpi"><div class="label">Cases (count)</div><div class="value" id="kpiCases">-</div></div>
      <div class="kpi"><div class="label">Eligible receivers (judges)</div><div class="value" id="kpiEligible">-</div></div>
      <div class="kpi"><div class="label">Moved cases (count)</div><div class="value" id="kpiMoved">-</div></div>
      <div class="kpi"><div class="label">Avg current (pred. days/case)</div><div class="value" id="kpiCurrent">-</div></div>
      <div class="kpi"><div class="label">Avg reassigned (pred. days/case)</div><div class="value" id="kpiReassigned">-</div></div>
      <div class="kpi"><div class="label">Predicted savings (total days; %)</div><div class="value" id="kpiSavings">-</div></div>
    </section>
    <div class="unit-note" id="scenarioNote">Time values are calendar days from hearing end to judgment. Predicted savings is the total predicted days saved across the filtered cases.</div>

    <section class="panel" style="margin:18px 0">
      <h2>Fixed-Effect Reconciliation</h2>
      <div class="quality" id="diagnostics"></div>
      <div class="unit-note" id="diagnosticNote"></div>
    </section>

    <section class="layout">
      <div class="panel">
        <h2>Top Recommended Assignment Moves</h2>
        <table id="actionsTable">
          <thead><tr><th>From judge</th><th>To judge</th><th class="number">Cases</th><th class="number">Savings (days)</th><th class="number">Avg savings (days/case)</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="panel">
        <h2>Current vs Reassigned</h2>
        <svg id="barChart" class="chart"></svg>
      </div>
    </section>

    <section class="grid2">
      <div class="panel">
        <h2>Turnaround Distribution</h2>
        <canvas id="histogram" width="640" height="300"></canvas>
      </div>
      <div class="panel">
        <h2>Judge Caseload vs Speed Effect</h2>
        <svg id="scatter" class="chart"></svg>
      </div>
    </section>

    <section class="grid2">
      <div class="panel">
        <h2>Judge Flow</h2>
        <table id="flowTable">
          <thead><tr><th>Judge</th><th class="number">Current</th><th class="number">New</th><th class="number">In</th><th class="number">Out</th><th class="number">Net</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="panel">
        <h2>Parse Quality</h2>
        <div class="quality" id="quality"></div>
        <table id="warningsTable">
          <thead><tr><th>Warning</th><th class="number">Count</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </section>

    <section class="panel" style="margin-top:18px">
      <h2>Cases With Largest Predicted Savings</h2>
      <table id="casesTable">
        <thead><tr><th>Case</th><th>Topic</th><th>From</th><th>To</th><th class="number">Current pred. days</th><th class="number">New pred. days</th><th class="number">Savings days</th></tr></thead>
        <tbody></tbody>
      </table>
    </section>
  </main>

  <script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
  <script>
    let DATA = null;
    const state = {};
    let renderQueued = false;

    const fmt = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : "-";
    const formatP = value => {
      if (!Number.isFinite(value)) return "-";
      if (value === 0) return "<1e-12";
      if (value < 0.001) return value.toExponential(2);
      return value.toFixed(3);
    };
    const pct = value => Number.isFinite(value) ? value.toFixed(1) + "%" : "-";
    const short = (text, n = 78) => (text || "").length > n ? (text || "").slice(0, n - 1) + "…" : (text || "");

    function bucket(row, mode) {
      if (mode === "month") return row.year_month || String(row.judgment_year || "");
      if (mode === "quarter") return `${row.judgment_year || ""}-Q${row.judgment_quarter || ""}`;
      return String(row.judgment_year || "");
    }

    function predFor(row, judgeId, modelView, effectByJudge) {
      const effect = effectByJudge.get(judgeId) || 0;
      if (modelView === "raw") return Math.max(0, (row.base_pred_raw || 0) + effect);
      return Math.max(0, Math.expm1((row.base_pred_log || 0) + effect));
    }

    function selectedRows() {
      const y0 = Number(document.getElementById("yearStart").value);
      const y1 = Number(document.getElementById("yearEnd").value);
      const court = document.getElementById("court").value;
      const topic = document.getElementById("topic").value;
      return DATA.cases.filter(row => {
        if (row.judgment_year < y0 || row.judgment_year > y1) return false;
        if (court !== "all" && row.court_division !== court) return false;
        if (topic !== "all" && !(row.topic_labels || []).includes(topic)) return false;
        return true;
      });
    }

    function runScenario() {
      const rows = selectedRows();
      const minCases = Math.max(1, Number(document.getElementById("minCases").value) || 1);
      const slack = Math.max(0, Number(document.getElementById("capacitySlack").value) || 0);
      const timeBucket = document.getElementById("timeBucket").value;
      const modelView = document.getElementById("modelView").value;
      const judgeRows = DATA.judges || [];
      const modelEligible = new Set(judgeRows.filter(j => j.eligible_for_assignment).map(j => j.judge_id));
      const effectByJudge = new Map(judgeRows.filter(j => j.eligible_for_assignment).map(j => [j.judge_id, modelView === "raw" ? (j.effect_raw_days || 0) : (j.effect_log || 0)]));
      const displayByJudge = new Map(judgeRows.map(j => [j.judge_id, j.judge || j.judge_display || j.judge_id]));
      for (const row of rows) displayByJudge.set(row.judge_id, row.judge);

      const globalCounts = new Map();
      for (const row of rows) globalCounts.set(row.judge_id, (globalCounts.get(row.judge_id) || 0) + 1);
      const eligibleGlobal = new Set([...globalCounts.entries()].filter(([judge, count]) => count >= minCases && modelEligible.has(judge)).map(([judge]) => judge));

      const grouped = new Map();
      for (const row of rows) {
        const key = bucket(row, timeBucket);
        if (!grouped.has(key)) grouped.set(key, []);
        grouped.get(key).push(row);
      }

      const assignments = [];
      for (const [key, bucketRows] of grouped.entries()) {
        const bucketCounts = new Map();
        for (const row of bucketRows) bucketCounts.set(row.judge_id, (bucketCounts.get(row.judge_id) || 0) + 1);
        const eligible = [...bucketCounts.keys()].filter(j => eligibleGlobal.has(j) && effectByJudge.has(j));
        const extra = new Map(eligible.map(j => [j, Math.max(0, Math.ceil((bucketCounts.get(j) || 0) * (1 + slack)) - (bucketCounts.get(j) || 0))]));

        const planned = bucketRows.map(row => {
          const currentPred = predFor(row, row.judge_id, modelView, effectByJudge);
          let bestJudge = null;
          let bestPred = currentPred;
          for (const candidate of eligible) {
            if (candidate === row.judge_id || (extra.get(candidate) || 0) <= 0) continue;
            const candidatePred = predFor(row, candidate, modelView, effectByJudge);
            if (candidatePred < bestPred) {
              bestPred = candidatePred;
              bestJudge = candidate;
            }
          }
          return { savings: currentPred - bestPred, row, bestJudge, bestPred, currentPred, bucket: key };
        }).sort((a, b) => b.savings - a.savings);

        for (const item of planned) {
          let toJudge = item.row.judge_id;
          let reassignedPred = item.currentPred;
          let moved = false;
          if (item.bestJudge && item.savings > 0.000001 && (extra.get(item.bestJudge) || 0) > 0) {
            toJudge = item.bestJudge;
            reassignedPred = item.bestPred;
            extra.set(item.bestJudge, (extra.get(item.bestJudge) || 0) - 1);
            moved = true;
          }
          assignments.push({
            row: item.row,
            bucket: item.bucket,
            from_judge_id: item.row.judge_id,
            to_judge_id: toJudge,
            from_judge: displayByJudge.get(item.row.judge_id) || item.row.judge_id,
            to_judge: displayByJudge.get(toJudge) || toJudge,
            moved,
            current_pred_days: item.currentPred,
            reassigned_pred_days: reassignedPred,
            predicted_savings_days: Math.max(0, item.currentPred - reassignedPred)
          });
        }
      }
      return summarize(assignments, rows, displayByJudge, eligibleGlobal.size, minCases);
    }

    function summarize(assignments, rows, displayByJudge, eligibleJudgeCount, minCases) {
      const currentTotal = assignments.reduce((sum, a) => sum + a.current_pred_days, 0);
      const newTotal = assignments.reduce((sum, a) => sum + a.reassigned_pred_days, 0);
      const moved = assignments.filter(a => a.moved);
      const actions = new Map();
      for (const a of moved) {
        const key = a.from_judge_id + "->" + a.to_judge_id;
        if (!actions.has(key)) actions.set(key, { from_judge: a.from_judge, to_judge: a.to_judge, cases: 0, savings: 0 });
        const action = actions.get(key);
        action.cases += 1;
        action.savings += a.predicted_savings_days;
      }
      const actionRows = [...actions.values()].map(a => ({ ...a, avg: a.savings / Math.max(1, a.cases) })).sort((a, b) => b.savings - a.savings);

      const currentCounts = new Map();
      const newCounts = new Map();
      const inflow = new Map();
      const outflow = new Map();
      for (const row of rows) currentCounts.set(row.judge_id, (currentCounts.get(row.judge_id) || 0) + 1);
      for (const a of assignments) {
        newCounts.set(a.to_judge_id, (newCounts.get(a.to_judge_id) || 0) + 1);
        if (a.moved) {
          inflow.set(a.to_judge_id, (inflow.get(a.to_judge_id) || 0) + 1);
          outflow.set(a.from_judge_id, (outflow.get(a.from_judge_id) || 0) + 1);
        }
      }
      const judgeIds = new Set([...currentCounts.keys(), ...newCounts.keys()]);
      const flowRows = [...judgeIds].map(j => ({
        judge: displayByJudge.get(j) || j,
        current: currentCounts.get(j) || 0,
        next: newCounts.get(j) || 0,
        inflow: inflow.get(j) || 0,
        outflow: outflow.get(j) || 0,
        net: (newCounts.get(j) || 0) - (currentCounts.get(j) || 0)
      })).sort((a, b) => Math.abs(b.net) - Math.abs(a.net) || (b.inflow + b.outflow) - (a.inflow + a.outflow));

      return {
        assignments,
        rows,
        actionRows,
        flowRows,
        summary: {
          n: assignments.length,
          moved: moved.length,
          avgCurrent: assignments.length ? currentTotal / assignments.length : 0,
          avgNew: assignments.length ? newTotal / assignments.length : 0,
          savings: currentTotal - newTotal,
          savingsPct: currentTotal > 0 ? (currentTotal - newTotal) / currentTotal * 100 : 0,
          eligibleJudges: eligibleJudgeCount,
          minCases
        }
      };
    }

    function render() {
      const scenario = runScenario();
      const s = scenario.summary;
      document.getElementById("kpiCases").textContent = String(s.n);
      document.getElementById("kpiEligible").textContent = String(s.eligibleJudges);
      document.getElementById("kpiMoved").textContent = `${s.moved}`;
      document.getElementById("kpiCurrent").textContent = fmt(s.avgCurrent);
      document.getElementById("kpiReassigned").textContent = fmt(s.avgNew);
      document.getElementById("kpiSavings").textContent = `${fmt(s.savings, 0)} (${pct(s.savingsPct)})`;
      document.getElementById("scenarioNote").textContent =
        `Units: calendar days from hearing end to judgment. Avg current/reassigned are predicted days per case. Predicted savings is total predicted days saved across the filtered cases. ` +
        `Eligible receiving judges: ${s.eligibleJudges}, requiring a fitted judge effect and at least ${s.minCases} filtered cases.`;
      renderActions(scenario.actionRows);
      renderFlow(scenario.flowRows);
      renderCases(scenario.assignments);
      renderBar(s.avgCurrent, s.avgNew);
      renderHistogram(scenario.assignments);
      renderScatter(scenario.rows);
      renderDiagnostics();
      renderQuality();
    }

    function scheduleRender() {
      if (renderQueued) return;
      renderQueued = true;
      window.requestAnimationFrame(() => {
        renderQueued = false;
        render();
      });
    }

    function setRows(tableId, rows, htmlBuilder) {
      document.querySelector(`#${tableId} tbody`).innerHTML = rows.map(htmlBuilder).join("");
    }

    function renderActions(rows) {
      setRows("actionsTable", rows.slice(0, 25), row => `
        <tr><td>${escapeHtml(row.from_judge)}</td><td>${escapeHtml(row.to_judge)}</td><td class="number">${row.cases}</td><td class="number">${fmt(row.savings)}</td><td class="number">${fmt(row.avg)}</td></tr>`);
    }

    function renderFlow(rows) {
      setRows("flowTable", rows.slice(0, 35), row => `
        <tr><td>${escapeHtml(row.judge)}</td><td class="number">${row.current}</td><td class="number">${row.next}</td><td class="number">${row.inflow}</td><td class="number">${row.outflow}</td><td class="number">${row.net > 0 ? "+" : ""}${row.net}</td></tr>`);
    }

    function renderCases(assignments) {
      const rows = assignments.filter(a => a.moved).sort((a, b) => b.predicted_savings_days - a.predicted_savings_days).slice(0, 50);
      setRows("casesTable", rows, a => `
        <tr>
          <td><a href="${escapeAttr(a.row.case_url)}" target="_blank" rel="noreferrer">${escapeHtml(a.row.case_id || short(a.row.title, 40))}</a><div class="small">${escapeHtml(short(a.row.title, 95))}</div></td>
          <td>${(a.row.topic_labels || []).map(t => `<span class="pill">${escapeHtml(t)}</span>`).join("")}</td>
          <td>${escapeHtml(a.from_judge)}</td><td>${escapeHtml(a.to_judge)}</td>
          <td class="number">${fmt(a.current_pred_days)}</td><td class="number">${fmt(a.reassigned_pred_days)}</td><td class="number">${fmt(a.predicted_savings_days)}</td>
        </tr>`);
    }

    function renderBar(currentAvg, newAvg) {
      const svg = document.getElementById("barChart");
      const w = svg.clientWidth || 500, h = 280, pad = 44;
      const maxVal = Math.max(currentAvg, newAvg, 1);
      const bars = [
        { label: "Current", value: currentAvg, color: "#2266aa" },
        { label: "Reassigned", value: newAvg, color: "#16817a" }
      ];
      svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
      svg.innerHTML = `<line x1="${pad}" y1="${h-pad}" x2="${w-pad}" y2="${h-pad}" stroke="#30343a"/>` + bars.map((bar, i) => {
        const bw = 86, gap = 70, x = pad + 80 + i * (bw + gap);
        const bh = (h - pad * 2) * bar.value / maxVal;
        const y = h - pad - bh;
        return `<rect x="${x}" y="${y}" width="${bw}" height="${bh}" fill="${bar.color}" rx="5"></rect>
          <text x="${x + bw/2}" y="${y - 8}" text-anchor="middle" font-size="15" font-weight="700">${fmt(bar.value)}</text>
          <text x="${x + bw/2}" y="${h - 16}" text-anchor="middle" font-size="13">${bar.label}</text>`;
      }).join("");
    }

    function renderHistogram(assignments) {
      const canvas = document.getElementById("histogram");
      const ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      const current = assignments.map(a => a.current_pred_days);
      const next = assignments.map(a => a.reassigned_pred_days);
      const maxVal = Math.max(...current, ...next, 1);
      const bins = 28;
      const hist = values => {
        const out = Array(bins).fill(0);
        for (const v of values) out[Math.min(bins - 1, Math.floor(v / maxVal * bins))] += 1;
        return out;
      };
      const h1 = hist(current), h2 = hist(next);
      const maxCount = Math.max(...h1, ...h2, 1);
      const pad = 34, bw = (w - pad * 2) / bins;
      ctx.strokeStyle = "#30343a";
      ctx.beginPath(); ctx.moveTo(pad, h - pad); ctx.lineTo(w - pad, h - pad); ctx.stroke();
      h1.forEach((count, i) => {
        const bh = (h - pad * 2) * count / maxCount;
        ctx.fillStyle = "rgba(34,102,170,.45)";
        ctx.fillRect(pad + i * bw, h - pad - bh, bw * .92, bh);
      });
      h2.forEach((count, i) => {
        const bh = (h - pad * 2) * count / maxCount;
        ctx.fillStyle = "rgba(22,129,122,.5)";
        ctx.fillRect(pad + i * bw + bw * .18, h - pad - bh, bw * .55, bh);
      });
      ctx.fillStyle = "#475467";
      ctx.font = "12px sans-serif";
      ctx.fillText("blue=current, green=reassigned predicted days", pad, 18);
      ctx.fillText("0", pad, h - 10);
      ctx.fillText(fmt(maxVal, 0), w - pad - 28, h - 10);
    }

    function renderScatter(rows) {
      const svg = document.getElementById("scatter");
      const w = svg.clientWidth || 500, h = 290, pad = 42;
      const counts = new Map();
      for (const row of rows) counts.set(row.judge_id, (counts.get(row.judge_id) || 0) + 1);
      const judges = (DATA.judges || []).filter(j => counts.has(j.judge_id)).map(j => ({
        judge: j.judge || j.judge_id,
        count: counts.get(j.judge_id) || 0,
        effect: document.getElementById("modelView").value === "raw" ? (j.effect_raw_days || 0) : (j.effect_log || 0)
      }));
      const maxCount = Math.max(...judges.map(j => j.count), 1);
      const effects = judges.map(j => j.effect);
      const minEff = Math.min(...effects, -0.01), maxEff = Math.max(...effects, 0.01);
      const x = c => pad + (w - pad * 2) * c / maxCount;
      const y = e => h - pad - (h - pad * 2) * (e - minEff) / (maxEff - minEff || 1);
      svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
      svg.innerHTML = `<line x1="${pad}" y1="${h-pad}" x2="${w-pad}" y2="${h-pad}" stroke="#30343a"/>
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h-pad}" stroke="#30343a"/>
        <text x="${w/2}" y="${h-8}" text-anchor="middle" font-size="12">Cases</text>
        <text x="12" y="${pad-14}" font-size="12">Effect</text>` + judges.map(j => {
          const color = j.effect <= 0 ? "#16817a" : "#c83b18";
          return `<circle cx="${x(j.count)}" cy="${y(j.effect)}" r="${Math.max(3, Math.min(9, Math.sqrt(j.count) + 2))}" fill="${color}" opacity=".72"><title>${escapeHtml(j.judge)} | cases=${j.count} | effect=${fmt(j.effect, 3)}</title></circle>`;
        }).join("");
    }

    function renderDiagnostics() {
      const modelView = document.getElementById("modelView").value;
      const diagnostic = DATA.model_stats?.fixed_effect_diagnostics?.[modelView] || {};
      const f = diagnostic.f_test || {};
      const d = diagnostic.dispersion || {};
      const effectDigits = modelView === "raw" ? 1 : 3;
      document.getElementById("diagnostics").innerHTML = [
        ["Target", diagnostic.target || "-"],
        ["Judge FE F-test", Number.isFinite(f.stat) ? fmt(f.stat, 2) : "-"],
        ["F-test p-value", formatP(f.p_value)],
        ["Effect std.", fmt(d.std, effectDigits)],
        ["P10 to P90", `${fmt(d.p10, effectDigits)} to ${fmt(d.p90, effectDigits)}`],
        ["Adj. R²", fmt(diagnostic.unrestricted_adj_r2, 3)]
      ].map(([k, v]) => `<div><div class="small">${escapeHtml(k)}</div><strong>${escapeHtml(v)}</strong></div>`).join("");
      document.getElementById("diagnosticNote").textContent =
        "This diagnostic is the original unregularized fixed-effect check: compare controls-only OLS against controls plus judge indicators. Reassignment still uses ridge-shrunk judge effects for more stable prediction.";
    }

    function renderQuality() {
      const q = DATA.parse_quality || {};
      document.getElementById("quality").innerHTML = [
        ["Raw", q.raw_cases || 0],
        ["Model-ready", q.model_ready_cases || 0],
        ["Rejected", q.rejected_cases || 0],
        ["Topics", Object.keys(q.topic_counts || {}).length]
      ].map(([k, v]) => `<div><div class="small">${k}</div><strong>${v}</strong></div>`).join("");
      const warnings = Object.entries(q.warning_counts || {}).sort((a, b) => b[1] - a[1]).slice(0, 18);
      setRows("warningsTable", warnings, ([name, count]) => `<tr><td>${escapeHtml(name)}</td><td class="number">${count}</td></tr>`);
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[c]));
    }
    function escapeAttr(value) { return escapeHtml(value).replace(/"/g, "&quot;"); }

    function initControls() {
      const years = DATA.years || [];
      const q = DATA.parse_quality || {};
      document.getElementById("dataNote").textContent =
        `Cached BAILII sample: ${q.raw_cases || 0} parsed cases, ${q.model_ready_cases || 0} model-ready cases. ` +
        "BAILII currently blocks additional crawling, so this dashboard is limited to the available cache. Raw judgment text is not embedded in this page.";
      const yearStart = document.getElementById("yearStart");
      const yearEnd = document.getElementById("yearEnd");
      yearStart.innerHTML = years.map(y => `<option value="${y}">${y}</option>`).join("");
      yearEnd.innerHTML = years.map(y => `<option value="${y}">${y}</option>`).join("");
      if (years.length) {
        yearStart.value = years[0];
        yearEnd.value = years[years.length - 1];
      }
      document.getElementById("court").innerHTML += (DATA.courts || []).map(c => `<option value="${c}">${c}</option>`).join("");
      document.getElementById("topic").innerHTML += (DATA.topics || []).map(t => `<option value="${escapeAttr(t)}">${escapeHtml(t)}</option>`).join("");
      document.getElementById("minCases").min = "1";
      document.getElementById("minCases").value = Number(DATA.settings?.min_judge_cases || 5);
      document.getElementById("capacitySlack").value = DATA.settings?.capacity_slack ?? 0.10;
      document.getElementById("timeBucket").value = DATA.settings?.time_bucket || "year";
      for (const id of ["yearStart", "yearEnd", "court", "topic", "minCases", "capacitySlack", "timeBucket", "modelView"]) {
        const el = document.getElementById(id);
        el.addEventListener("input", scheduleRender);
        el.addEventListener("change", scheduleRender);
        el.addEventListener("keyup", scheduleRender);
        el.addEventListener("blur", render);
      }
      document.getElementById("applyFilters").addEventListener("click", render);
    }

    try {
      DATA = JSON.parse(document.getElementById("dashboard-data").textContent);
      initControls();
      render();
    } catch (err) {
      console.error(err);
      document.body.innerHTML = "<main><h1>Failed to load embedded dashboard data</h1><p>Run <code>python v2.py</code> to regenerate the static dashboard.</p></main>";
    }
  </script>
</body>
</html>
"""
    html_doc = html_doc.replace("__DASHBOARD_DATA__", embedded_data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)


# -----------------------------
# Main
# -----------------------------


def is_notebook_runtime() -> bool:
    return (
        "google.colab" in sys.modules
        or "ipykernel" in sys.modules
        or bool(os.environ.get("JPY_PARENT_PID"))
    )


def has_notebook_kernel_args(argv: Optional[Sequence[str]] = None) -> bool:
    args = list(sys.argv[1:] if argv is None else argv)
    return any(
        arg == "-f"
        or arg.endswith(".json") and ("/runtime/kernel-" in arg or "\\runtime\\kernel-" in arg)
        or "colab_kernel_launcher.py" in arg
        for arg in args
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BAILII KB/QB judge reassignment analysis")
    parser.add_argument("--max-cases", default="all", help="Number of cases to use, or 'all' (default: all)")
    parser.add_argument("--court-scope", choices=["kb-qb", "kb"], default="kb-qb")
    parser.add_argument("--metric", choices=["reserved_days"], default="reserved_days")
    parser.add_argument("--min-judge-cases", type=int, default=DEFAULT_MIN_JUDGE_CASES)
    parser.add_argument("--capacity-slack", type=float, default=DEFAULT_CAPACITY_SLACK)
    parser.add_argument("--time-bucket", choices=["year", "quarter", "month"], default=DEFAULT_TIME_BUCKET)
    parser.add_argument("--refresh-parse", action="store_true", help="Refetch cached URLs from BAILII instead of using cached text")
    if argv is None and is_notebook_runtime():
        args, ignored = parser.parse_known_args(sys.argv[1:])
        if ignored:
            print(f"[notebook] ignoring kernel args: {' '.join(ignored)}")
        return args
    return parser.parse_args(argv)


def run_colab(
    max_cases: str = "all",
    refresh_parse: bool = True,
    court_scope: str = "kb-qb",
    min_judge_cases: int = DEFAULT_MIN_JUDGE_CASES,
    capacity_slack: float = DEFAULT_CAPACITY_SLACK,
    time_bucket: str = DEFAULT_TIME_BUCKET,
) -> int:
    """Notebook-friendly entrypoint after pasting this file into Colab."""
    argv = [
        "--max-cases", str(max_cases),
        "--court-scope", court_scope,
        "--metric", "reserved_days",
        "--min-judge-cases", str(min_judge_cases),
        "--capacity-slack", str(capacity_slack),
        "--time-bucket", time_bucket,
    ]
    if refresh_parse:
        argv.append("--refresh-parse")
    return main(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    max_cases = parse_max_cases(args.max_cases)
    if args.min_judge_cases < 1:
        raise SystemExit("--min-judge-cases must be at least 1")
    if args.capacity_slack < 0:
        raise SystemExit("--capacity-slack must be nonnegative")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_cases = load_cases(CASES_RAW_PATH)
    session = requests.Session()

    print("=== CASE INGESTION ===")
    print(f"Source: BAILII | court_scope={args.court_scope} | max_cases={args.max_cases}")
    try:
        cases = crawl_cases(
            session=session,
            max_cases=max_cases,
            seed_cases=seed_cases,
            refresh_parse=args.refresh_parse,
            court_scope=args.court_scope,
        )
    except BailiiBlockedError as exc:
        print(f"[blocked] {exc}", file=sys.stderr)
        return 2

    save_json(CASES_RAW_PATH, cases)
    print(f"Accepted into raw cache: {len(cases)}")

    print("=== VALIDATION ===")
    speed_rows, rejections, parse_quality = build_speed_dataset(cases)
    save_json(CRAWL_REJECTIONS_PATH, rejections)
    if len(speed_rows) < 30:
        print(f"Too few model-ready cases ({len(speed_rows)}). See {CRAWL_REJECTIONS_PATH}.", file=sys.stderr)
        return 1
    print(f"Model-ready cases: {len(speed_rows)} | rejected: {len(rejections)}")

    aliases = build_judge_aliases(cases)
    save_json(JUDGE_ALIASES_PATH, aliases)

    print("=== MODEL ===")
    for row in speed_rows:
        row["reserved_log"] = math.log1p(row["reserved_days"])
    plan = build_feature_plan(speed_rows, args.min_judge_cases)
    log_model = fit_ridge(speed_rows, "reserved_log", plan)
    raw_model = fit_ridge(speed_rows, "reserved_days", plan)
    add_model_predictions(speed_rows, plan, log_model, raw_model)
    judge_rows = build_judge_effect_rows(speed_rows, aliases, log_model, raw_model)
    fixed_effect_diagnostics = run_fixed_effect_diagnostics(speed_rows, plan)

    assignment_results = run_assignment(
        speed_rows,
        log_model["judge_effects"],
        min_judge_cases=args.min_judge_cases,
        capacity_slack=args.capacity_slack,
        time_bucket=args.time_bucket,
        model_view="log",
    )

    save_json(CASES_OUTPUT_PATH, speed_rows)
    save_json(JUDGE_EFFECTS_PATH, judge_rows)
    save_json(ASSIGNMENT_RESULTS_PATH, assignment_results)

    t_summary = summarize_turnaround(speed_rows)
    speed_results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_cases": len(speed_rows),
        "n_judges": len(judge_rows),
        "metric": args.metric,
        "controls": CATEGORY_FEATURES + NUMERIC_FEATURES,
        "min_judge_cases": args.min_judge_cases,
        "capacity_slack": args.capacity_slack,
        "time_bucket": args.time_bucket,
        "turnaround_summary": t_summary,
        "model": {
            "log": {"r2": log_model["r2"], "rmse": log_model["rmse"], "target": "log1p(reserved_days)"},
            "raw": {"r2": raw_model["r2"], "rmse": raw_model["rmse"], "target": "reserved_days"},
            "eligible_judges": len(plan.eligible_judges),
            "ridge_alpha": RIDGE_ALPHA,
            "fixed_effect_diagnostics": fixed_effect_diagnostics,
        },
        "assignment_summary": assignment_results["summary"],
        "parse_quality": parse_quality,
        "outputs": {
            "cases_raw": str(CASES_RAW_PATH),
            "cases_speed": str(CASES_OUTPUT_PATH),
            "judge_effects": str(JUDGE_EFFECTS_PATH),
            "assignment_results": str(ASSIGNMENT_RESULTS_PATH),
            "dashboard_data": str(DASHBOARD_DATA_PATH),
            "crawl_rejections": str(CRAWL_REJECTIONS_PATH),
            "judge_aliases": str(JUDGE_ALIASES_PATH),
            "dashboard": str(INDEX_PATH),
        },
    }
    save_json(SPEED_RESULTS_PATH, speed_results)

    dashboard_data = build_dashboard_data(
        speed_rows,
        judge_rows,
        assignment_results,
        parse_quality,
        log_model,
        raw_model,
        fixed_effect_diagnostics,
        args,
    )
    save_json(DASHBOARD_DATA_PATH, dashboard_data)
    write_dashboard_html(INDEX_PATH, dashboard_data)

    print("=== RESULTS ===")
    print(json.dumps({
        "model_ready_cases": len(speed_rows),
        "judges": len(judge_rows),
        "eligible_judges": len(plan.eligible_judges),
        "log_r2": round(log_model["r2"], 4),
        "raw_r2": round(raw_model["r2"], 4),
        "fixed_effect_log_p_value": fixed_effect_diagnostics["log"]["f_test"]["p_value"],
        "moved_cases": assignment_results["summary"]["moved_cases"],
        "predicted_savings_days": round(assignment_results["summary"]["predicted_savings_days"], 1),
        "predicted_savings_pct": round(assignment_results["summary"]["predicted_savings_pct"], 2),
    }, indent=2))
    print(f"Saved dashboard: {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    if is_notebook_runtime() and has_notebook_kernel_args():
        print("Notebook kernel detected; v2.py was loaded but not auto-run.")
        print("Run this in a new cell when ready:")
        print("  run_colab(max_cases='all', refresh_parse=True)")
    else:
        raise SystemExit(main())
