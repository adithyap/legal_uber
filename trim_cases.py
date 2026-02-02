#!/usr/bin/env python3
"""
Trim a cases_raw.json down to the small set of fields needed for report.html / sharing.

Usage:
  python trim_cases.py --input cases_raw.json --output cases_trimmed.json

Input format: either a list of case dicts, or {"cases": [...]}.

Keeps only lightweight fields:
  case_id, title, judgment_date, area_of_law, case_url, opinion_author, justices, hearing_start, hearing_end
Everything else (full text, analysis_text, etc.) is dropped.
"""
import argparse
import json
from typing import List, Dict, Any

KEEP_KEYS = {
    "case_id",
    "title",
    "judgment_date",
    "area_of_law",
    "case_url",
    "opinion_author",
    "justices",
    "hearing_start",
    "hearing_end",
}

def load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return data["cases"]
    raise ValueError("Input JSON must be a list or contain a 'cases' array")


def trim_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trimmed = []
    for c in cases:
        out = {k: c.get(k) for k in KEEP_KEYS if k in c}
        trimmed.append(out)
    return trimmed


def main():
    ap = argparse.ArgumentParser(description="Trim cases_raw.json to minimal report fields")
    ap.add_argument("--input", required=True, help="Path to cases_raw.json")
    ap.add_argument("--output", required=True, help="Path to write trimmed JSON")
    args = ap.parse_args()

    cases = load_cases(args.input)
    trimmed = trim_cases(cases)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=2)

    print(f"Trimmed {len(cases)} cases -> {len(trimmed)} cases")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
