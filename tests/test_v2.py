import math
import tempfile
import unittest
from pathlib import Path

import v2


class HearingDateParsingTests(unittest.TestCase):
    def assert_dates(self, raw, judgment, expected_start, expected_end):
        start, end, warnings = v2.parse_hearing_dates(raw, judgment)
        self.assertEqual(start, expected_start, warnings)
        self.assertEqual(end, expected_end, warnings)

    def test_same_month_dash_range(self):
        self.assert_dates("Hearing dates: 26-27 May 2022", "2022-09-23", "2022-05-26", "2022-05-27")

    def test_same_month_and_list(self):
        self.assert_dates("Hearing dates: 11 and 16 April 2024;", "2024-06-07", "2024-04-11", "2024-04-16")

    def test_cross_month_range(self):
        self.assert_dates("Hearing dates: 31 October - 1 November 2022", "2022-12-01", "2022-10-31", "2022-11-01")

    def test_ordinal_ampersand_dates(self):
        self.assert_dates("Hearing dates: 9th & 10th May 2022", "2022-10-17", "2022-05-09", "2022-05-10")

    def test_complex_month_groups(self):
        self.assert_dates(
            "Hearing dates: 25, 26, 28 January, 1 - 4 February and 19 - 22 July 2022",
            "2022-10-10",
            "2022-01-25",
            "2022-07-22",
        )

    def test_semicolon_separated_hearing_blocks(self):
        self.assert_dates(
            "Hearing Dates: 2, 3, 6 and 7 February 2023; 22 and 23 June 2023; 1, 2 and 5 February 2024; 7 March 2024",
            "2024-10-06",
            "2023-02-02",
            "2024-03-07",
        )

    def test_written_submission_clause_is_not_hearing_end(self):
        self.assert_dates(
            "Hearing dates: 11 and 16 April 2024; further written submissions 17 April 2024",
            "2024-06-07",
            "2024-04-11",
            "2024-04-16",
        )

    def test_malformed_hearing_line(self):
        start, end, warnings = v2.parse_hearing_dates("Hearing date:", "2022-10-10")
        self.assertIsNone(start)
        self.assertIsNone(end)
        self.assertIn("unable_to_parse_hearing_dates", warnings)


class JudgeCanonicalizationTests(unittest.TestCase):
    def test_titles_and_punctuation_normalize(self):
        self.assertEqual(v2.judge_id_from_name(": MRS JUSTICE COCKERILL DBE"), "mrs_justice_cockerill")
        self.assertEqual(v2.judge_id_from_name("Mrs Justice Cockerill:"), "mrs_justice_cockerill")

    def test_sitting_as_phrase_removed(self):
        self.assertEqual(
            v2.judge_id_from_name("HHJ Karen Walden-Smith sitting as a Judge of the High Court"),
            "hhj_karen_walden_smith",
        )

    def test_honourable_prefix_removed(self):
        self.assertEqual(
            v2.judge_id_from_name("THE HONOURABLE MR JUSTICE SWEETING"),
            "mr_justice_sweeting",
        )


class ValidationTests(unittest.TestCase):
    def test_title_judgment_date_overrides_bad_cached_date(self):
        lines = [
            "Example v Example [2024] EWHC 1668 (KB) (01 July 2024)",
            "Hearing dates: 12 and 13 June 2024",
        ]
        parsed = v2.extract_judgment_date(
            lines,
            "Example v Example [2024] EWHC 1668 (KB) (01 July 2024)",
            existing="2026-07-01",
        )
        self.assertEqual(parsed, "2024-07-01")

    def test_model_ready_case_requires_key_fields(self):
        case = {
            "case_url": "https://www.bailii.org/ew/cases/EWHC/KB/2024/1352.html",
            "case_id": "[2024] EWHC 1352 (KB)",
            "neutral_citation": "[2024] EWHC 1352 (KB)",
            "title": "Example",
            "judge_display": "MR JUSTICE EXAMPLE",
            "judge_id": "mr_justice_example",
            "judgment_date": "2024-06-07",
            "hearing_end": "2024-04-16",
            "reserved_days": 52,
            "court_division": "KB",
            "topic_labels": ["personal injury"],
        }
        ok, reasons = v2.validate_case(case)
        self.assertTrue(ok, reasons)
        rows, rejections, _ = v2.build_speed_dataset([case])
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rejections), 0)
        self.assertEqual(rows[0]["reserved_days"], 52)

    def test_negative_reserved_days_rejected(self):
        case = {
            "case_url": "https://www.bailii.org/ew/cases/EWHC/KB/2024/1.html",
            "case_id": "[2024] EWHC 1 (KB)",
            "neutral_citation": "[2024] EWHC 1 (KB)",
            "judge_display": "MR JUSTICE EXAMPLE",
            "judge_id": "mr_justice_example",
            "judgment_date": "2024-01-01",
            "hearing_end": "2024-01-02",
            "reserved_days": -1,
            "court_division": "KB",
        }
        ok, reasons = v2.validate_case(case)
        self.assertFalse(ok)
        self.assertIn("negative_reserved_days", reasons)


class FixedEffectDiagnosticTests(unittest.TestCase):
    def test_joint_judge_effect_f_test_detects_dispersion(self):
        rows = []
        for judge_id, judge_name, base_days in [
            ("judge_a", "Judge A", 5),
            ("judge_b", "Judge B", 12),
            ("judge_c", "Judge C", 31),
        ]:
            for idx in range(5):
                days = base_days + (idx % 2)
                rows.append({
                    "case_id": f"{judge_id}-{idx}",
                    "judge_id": judge_id,
                    "judge_display": judge_name,
                    "judgment_year": 2024,
                    "judgment_month": 1,
                    "judgment_quarter": 1,
                    "year_month": "2024-01",
                    "court_division": "KB",
                    "topic_labels": ["contract"],
                    "primary_topic": "contract",
                    "reserved_days": days,
                    "reserved_log": math.log1p(days),
                    "hearing_span_days": 1,
                    "text_char_count": 1000,
                    "word_count": 180,
                    "paragraph_count": 20,
                })

        plan = v2.build_feature_plan(rows, min_judge_cases=1)
        diagnostics = v2.run_fixed_effect_diagnostics(rows, plan)

        raw = diagnostics["raw"]
        self.assertEqual(raw["f_test"]["df_num"], 2)
        self.assertLess(raw["f_test"]["p_value"], 0.001)
        self.assertGreater(raw["dispersion"]["range"], 20.0)


class DashboardTests(unittest.TestCase):
    def test_dashboard_html_embeds_compact_data_without_fetching_json(self):
        dashboard_data = {
            "generated_at": "2026-05-14T00:00:00Z",
            "settings": {"min_judge_cases": 1, "capacity_slack": 0.1, "time_bucket": "year", "metric": "reserved_days"},
            "years": [2024],
            "courts": ["KB"],
            "topics": ["contract"],
            "cases": [],
            "judges": [],
            "parse_quality": {"raw_cases": 0, "model_ready_cases": 0, "rejected_cases": 0, "warning_counts": {}, "topic_counts": {}},
            "model_stats": {"fixed_effect_diagnostics": {}},
            "assignment_results": {"summary": {}, "top_actions": [], "judge_flow": []},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "index.html"
            v2.write_dashboard_html(path, dashboard_data)
            html = path.read_text(encoding="utf-8")

        self.assertIn('<script id="dashboard-data" type="application/json">', html)
        self.assertIn('"generated_at":"2026-05-14T00:00:00Z"', html)
        self.assertIn("How to read the controls", html)
        self.assertIn("Capacity slack", html)
        self.assertNotIn("control-help", html)
        self.assertNotIn("BAILII Court Allocation", html)
        self.assertNotIn('fetch("dashboard_data.json")', html)


if __name__ == "__main__":
    unittest.main()
