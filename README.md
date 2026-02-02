# Judge assignment + sharing utilities

## v2.py
- End-to-end pipeline for UK court cases (Supreme Court or EWHC/KB), from crawl to judge-assignment evaluation.
- Crawls cases, parses metadata/text, builds embeddings/keywords/complexity, trains judge profiles, scores test cases, and writes outputs (assignments, reports, profiles) into the cache/export folder.
- Configurable via top-level flags (e.g., `COURT_MODE`, `PANEL_JUDGES_ONLY`, `ZIP_RESULTS`).
- Automatically zips export artifacts for Colab download.

## trim_cases.py
- Lightweight helper to shrink `cases_raw.json` into a minimal `cases_raw_trimmed.json` for sharing/reporting.
- Keeps only: `case_id`, `title`, `judgment_date`, `area_of_law`, `case_url`, `opinion_author`, `justices`, `hearing_start`, `hearing_end`.
- Usage: `python trim_cases.py --input cases_raw.json --output cases_trimmed.json`.
