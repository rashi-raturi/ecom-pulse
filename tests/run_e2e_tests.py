"""
Automated E2E Test Suite Runner for Lakehouse Ecommerce Data Pipeline.

Executes 4-tier systematic test suite and outputs structured scorecard:
- Constraints Validation (R3 / F22)
- Tier 1: Feature Coverage (F01, F02, F05, F06, F11, F15, F16, F21)
- Tier 2: Boundary & Corner Cases (F04, Nulls, Div-by-Zero, Timestamps, Hash Safety)
- Tier 3: Cross-Feature Combinations (F03, F07, F08, F10, F12, F13, F14, F17)
- Tier 4: Real-World Scenarios (F18, F19, F20, Oracles, Lineage Contracts)

Usage:
    python tests/run_e2e_tests.py
    python tests/run_e2e_tests.py --tier 1
    python tests/run_e2e_tests.py --tier constraints
    python tests/run_e2e_tests.py --verbose
    python tests/run_e2e_tests.py --json
"""

import argparse
import io
import json
import os
import sys
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests.test_e2e_lakehouse import (
    TestConstraintsValidation,
    TestTier1FeatureCoverage,
    TestTier2BoundaryAndCornerCases,
    TestTier3CrossFeatureCombinations,
    TestTier4RealWorldScenarios,
)


TIER_MAPPING = {
    "constraints": {
        "name": "Constraints Validation (R3 / F22: Scope Exclusions)",
        "test_class": TestConstraintsValidation,
    },
    "1": {
        "name": "Tier 1: Feature Coverage (F01, F02, F05, F06, F11, F15, F16, F21)",
        "test_class": TestTier1FeatureCoverage,
    },
    "2": {
        "name": "Tier 2: Boundary & Corner Cases (F04, Nulls, Div-by-Zero, Timestamps, Hash)",
        "test_class": TestTier2BoundaryAndCornerCases,
    },
    "3": {
        "name": "Tier 3: Cross-Feature Combinations (F03, F07, F08, F10, F12, F13, F14, F17)",
        "test_class": TestTier3CrossFeatureCombinations,
    },
    "4": {
        "name": "Tier 4: Real-World Scenarios (F18, F19, F20, Oracles, Lineage Contracts)",
        "test_class": TestTier4RealWorldScenarios,
    },
}


class TierResultTracker:
    def __init__(self, tier_id: str, tier_name: str):
        self.tier_id = tier_id
        self.tier_name = tier_name
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []
        self.errors: List[Tuple[str, str]] = []
        self.duration: float = 0.0

    @property
    def total(self) -> int:
        return len(self.passed) + len(self.failed) + len(self.errors)

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 100.0
        return (len(self.passed) / self.total) * 100.0


def run_tier(tier_id: str, verbose: bool = False) -> TierResultTracker:
    tier_info = TIER_MAPPING[tier_id]
    tracker = TierResultTracker(tier_id, tier_info["name"])
    test_class = tier_info["test_class"]
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(test_class)
    
    start_time = time.time()
    for test in suite:
        test_name = test._testMethodName
        # Create single test runner with buffer
        result = unittest.TestResult()
        test.run(result)
        
        if result.wasSuccessful():
            tracker.passed.append(test_name)
            if verbose:
                print(f"  [PASS] {test_name}")
        elif result.failures:
            _, traceback_str = result.failures[0]
            tracker.failed.append((test_name, traceback_str))
            if verbose:
                print(f"  [FAIL] {test_name}")
        elif result.errors:
            _, traceback_str = result.errors[0]
            tracker.errors.append((test_name, traceback_str))
            if verbose:
                print(f"  [ERROR] {test_name}")
                
    tracker.duration = time.time() - start_time
    return tracker


def get_remediation_hint(test_name: str, error_msg: str) -> str:
    """Provide specific remediation guidance based on failed test and error."""
    if "f01_bronze_products_ingestion" in test_name:
        return "Milestone 1 Action: In 'bronze/bronze_ingestion.ipynb', add CSV ingestion for 'olist_products_dataset.csv' writing to 'workspace.ecompulse_bronze.products' in Delta format."
    elif "f02_gold_notebooks_exist" in test_name:
        return "Milestone 1/2 Action: Create missing gold notebooks under 'gold/': gold_sales_analytics.ipynb, gold_customer_analytics.ipynb, gold_product_analytics.ipynb, gold_scd2_dimensions.ipynb."
    elif "f04_sales_fan_out" in test_name:
        return "Milestone 1 Action: In 'gold/gold_sales_analytics.ipynb', ensure order_items and payments are pre-aggregated (grouped by order_id) BEFORE joining with orders."
    elif "growth_rate_division_by_zero" in test_name:
        return "Milestone 1 Action: In 'gold/gold_sales_analytics.ipynb', wrap revenue growth rate denominator with F.when(prev != 0, ...) to prevent division by zero."
    elif "f07_sales_window" in test_name:
        return "Milestone 1 Action: In 'gold/gold_sales_analytics.ipynb', use PySpark Window for running total revenue, 7-day moving average, and MoM growth."
    elif "f08_f09_f10_customer" in test_name:
        return "Milestone 1 Action: In 'gold/gold_customer_analytics.ipynb', compute customer LTV, basket size, row_number(), dense_rank(), and ntile(10)."
    elif "f12_f13_f14_product" in test_name:
        return "Milestone 1 Action: In 'gold/gold_product_analytics.ipynb', compute product sales aggregations, dense_rank() by category, and Pareto 80/20 category share."
    elif "f17_scd2_delta_merge" in test_name:
        return "Milestone 2 Action: In 'gold/gold_scd2_dimensions.ipynb', implement DeltaTable staged MERGE on customer_id and is_current, with whenMatchedUpdate and whenNotMatchedInsert."
    elif "f18_incremental" in test_name:
        return "Milestone 2 Action: In 'gold/gold_scd2_dimensions.ipynb', demonstrate incremental loading using watermark or timestamp filtering."
    elif "f19_schema" in test_name:
        return "Milestone 2 Action: In 'gold/gold_scd2_dimensions.ipynb', demonstrate schema enforcement and evolution with mergeSchema."
    elif "f20_delta_time_travel" in test_name:
        return "Milestone 2 Action: In 'gold/gold_scd2_dimensions.ipynb', include time travel queries (VERSION AS OF / TIMESTAMP AS OF / DESCRIBE HISTORY)."
    elif "f21_sql_preview" in test_name:
        return "Milestone 1/2 Action: Ensure each Gold notebook ends with at least one terminal %sql preview cell."
    elif "forbidden" in error_msg.lower() or "kafka" in test_name or "airflow" in test_name or "postgres" in test_name or "streaming" in test_name:
        return "Constraint Action: Remove forbidden external service imports/configs (Kafka, Airflow, Postgres, Spark Streaming)."
    else:
        # Extract first line of assertion failure
        lines = error_msg.strip().splitlines()
        last_line = lines[-1] if lines else "Check test assertion"
        return f"Implementation Action: {last_line}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Lakehouse E2E Test Suite")
    parser.add_argument(
        "--tier",
        choices=["all", "constraints", "1", "2", "3", "4"],
        default="all",
        help="Specific tier to execute (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed test case execution",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )
    args = parser.parse_args()

    selected_tiers = list(TIER_MAPPING.keys()) if args.tier == "all" else [args.tier]
    
    if not args.json:
        print("=" * 80)
        print(" ECOMMERCE DATA PIPELINE — LAKEHOUSE E2E TEST SUITE")
        print(f" Environment: Python {sys.version.split()[0]} | Working Dir: {REPO_ROOT}")
        print("=" * 80)

    trackers: List[TierResultTracker] = []
    
    for tier_id in selected_tiers:
        tier_info = TIER_MAPPING[tier_id]
        if not args.json:
            print(f"\n>> Executing {tier_info['name']}...")
        tracker = run_tier(tier_id, verbose=args.verbose)
        trackers.append(tracker)
        if not args.json and not args.verbose:
            status_symbol = "[OK]" if (len(tracker.failed) == 0 and len(tracker.errors) == 0) else "[FAILURES]"
            print(f"   {status_symbol} {len(tracker.passed)} passed, {len(tracker.failed)} failed, {len(tracker.errors)} errors ({tracker.duration:.2f}s)")

    # Aggregate metrics
    total_tests = sum(t.total for t in trackers)
    total_passed = sum(len(t.passed) for t in trackers)
    total_failed = sum(len(t.failed) for t in trackers)
    total_errors = sum(len(t.errors) for t in trackers)
    overall_pass_rate = (total_passed / total_tests * 100.0) if total_tests > 0 else 0.0

    if args.json:
        output_data = {
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "errors": total_errors,
                "pass_rate_pct": round(overall_pass_rate, 2),
            },
            "tiers": [
                {
                    "tier_id": t.tier_id,
                    "tier_name": t.tier_name,
                    "total": t.total,
                    "passed": len(t.passed),
                    "failed": len(t.failed),
                    "errors": len(t.errors),
                    "duration_s": round(t.duration, 3),
                    "failures": [{"test": name, "traceback": tb} for name, tb in t.failed],
                    "errors_list": [{"test": name, "traceback": tb} for name, tb in t.errors],
                }
                for t in trackers
            ]
        }
        print(json.dumps(output_data, indent=2))
        return 0 if (total_failed == 0 and total_errors == 0) else 1

    # Print Scorecard Table
    print("\n" + "=" * 80)
    print(" SCORECARD SUMMARY")
    print("=" * 80)
    print(f"{'Tier':<45} | {'Total':<6} | {'Pass':<5} | {'Fail':<5} | {'Rate':<7}")
    print("-" * 80)
    for t in trackers:
        short_name = t.tier_name.split(":")[0] if ":" in t.tier_name else t.tier_name[:40]
        print(f"{short_name:<45} | {t.total:<6} | {len(t.passed):<5} | {len(t.failed) + len(t.errors):<5} | {t.pass_rate:>5.1f}%")
    print("-" * 80)
    print(f"{'OVERALL':<45} | {total_tests:<6} | {total_passed:<5} | {total_failed + total_errors:<5} | {overall_pass_rate:>5.1f}%")
    print("=" * 80)

    # Print Failure Details and Actionable Remediation
    all_issues = []
    for t in trackers:
        for name, tb in t.failed:
            all_issues.append((t.tier_name, name, "FAIL", tb))
        for name, tb in t.errors:
            all_issues.append((t.tier_name, name, "ERROR", tb))

    if all_issues:
        print("\n" + "!" * 80)
        print(f" ACTIONABLE REMEDIATION FOR {len(all_issues)} FAILING TEST(S)")
        print("!" * 80)
        for idx, (tier_name, test_name, status, tb) in enumerate(all_issues, 1):
            hint = get_remediation_hint(test_name, tb)
            print(f"\n{idx}. [{status}] {test_name}")
            print(f"   Tier: {tier_name}")
            print(f"   Guidance: {hint}")
            if args.verbose:
                print("   Traceback snippet:")
                for line in tb.strip().splitlines()[-4:]:
                    print(f"     {line}")
        print("\n" + "!" * 80)
    else:
        print("\n>> All test tiers passed with 100% success!")

    return 0 if (total_failed == 0 and total_errors == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
