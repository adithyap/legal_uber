# Judge Reassignment Analysis (BAILII EWHC KB/QB)

This project analyzes whether England and Wales High Court King's/Queen's Bench cases might have moved faster under different judge assignments.

The main pipeline is:

```bash
python3 -m pip install -r requirements.txt
python3 v2.py --max-cases all --court-scope kb-qb
```

Useful cached-data run, avoiding a fresh BAILII crawl:

```bash
python3 v2.py --max-cases 1103 --court-scope kb-qb
```

## Colab

If you paste `v2.py` into a Colab cell, it will define the functions without auto-running against Colab's kernel arguments. Then run:

```python
!pip install beautifulsoup4 numpy python-dateutil requests tqdm
run_colab(max_cases="all", refresh_parse=True)
```

For a small smoke test first:

```python
run_colab(max_cases="20", refresh_parse=True)
```

## Method

- Source: BAILII only, covering modern `EWHC/KB` and historical `EWHC/QB`.
- Primary speed metric: `reserved_days = judgment_date - hearing_end`.
- Parsing validates required model fields and quarantines unusable cases in `data/crawl_rejections.json`.
- Topic labels are curated phrase/rule labels, not word-cloud terms.
- Model: regularized judge effects on `log1p(reserved_days)`, with raw-days robustness.
- Counterfactual: reassigns cases only within the selected time bucket and only up to each receiving judge's capacity slack.

## Outputs

- `data/cases_raw.json`: parsed raw cache; local only and not needed for GitHub Pages.
- `data/cases_speed.json`: model-ready case rows; generated local analysis output.
- `data/judge_effects.json`: canonical judge effects and aliases; generated local analysis output.
- `data/assignment_results.json`: default counterfactual assignment result; generated local analysis output.
- `data/dashboard_data.json`: compact local payload used to build the static dashboard.
- `data/crawl_rejections.json`: rejected cases and reasons.
- `data/index.html`: self-contained interactive dashboard with the compact display data embedded; this is the file to publish under GitHub Pages.

Run tests with:

```bash
python3 -m unittest tests/test_v2.py
```
