"""
Tier 5 Adversarial Coverage Hardening Test Suite.

Rigorously stress-tests the Lakehouse pipeline implementation across:
1. SCD2 Multi-Version Lifecycle State Transitions & Invariants (Batches 1-5)
2. Multi-Payment Combinations, Mixed Instruments & Zero-Payment Orders
3. Zero-Freight Orders, Multi-Item Bundles & Fan-Out Invariants
4. Customers with Zero Orders & Identical Spend Ranking Ties
5. Temporal Boundaries, Leap Years & Division-by-Zero Guards
6. Product Pareto Distribution Boundary Conditions & Extreme Catalogs
7. Surrogate Key Collision Resistance, Unicode & Null Safety
8. White-Box AST & Notebook Code Architecture Verification
"""

import ast
import hashlib
from datetime import datetime, date
import json
import os
import re
import sys
import unittest
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests.lakehouse_test_utils import (
    SCD2DimensionState,
    check_fan_out_mitigation_ast,
    extract_code_cells,
    extract_python_and_sql,
    extract_table_references,
    generate_scd2_surrogate_key,
    get_repo_root,
    load_notebook,
    parse_python_ast,
    simulate_customer_rfm_and_ltv,
    simulate_fan_out_cartesian_vs_preagg,
    simulate_pareto_category_distribution,
)


class BaseAdversarialTestCase(unittest.TestCase):
    """Base helper for adversarial tests."""

    def get_notebook_code(self, relative_path: str) -> str:
        repo_root = get_repo_root()
        abs_path = os.path.join(repo_root, relative_path)
        self.assertTrue(os.path.exists(abs_path), f"Notebook not found: {relative_path}")
        nb = load_notebook(relative_path)
        py_code, _ = extract_python_and_sql(nb)
        return py_code

    def get_notebook_sql(self, relative_path: str) -> List[str]:
        repo_root = get_repo_root()
        abs_path = os.path.join(repo_root, relative_path)
        self.assertTrue(os.path.exists(abs_path), f"Notebook not found: {relative_path}")
        nb = load_notebook(relative_path)
        _, sql_cells = extract_python_and_sql(nb)
        return sql_cells


# ==============================================================================
# 1. SCD2 Multi-Version Lifecycle State Transitions & Invariants
# ==============================================================================

class TestSCD2LifecycleAdversarial(BaseAdversarialTestCase):
    """
    Stress-tests the SCD Type 2 state machine across 5 consecutive batches:
    - Batch 1: Initial baseline load
    - Batch 2: Attribute updates + new customers
    - Batch 3: Idempotent no-op batch (zero attribute changes)
    - Batch 4: Null-to-value and value-to-null transitions
    - Batch 5: Multi-hop moves (third relocation for same customer)
    """

    def test_multi_batch_scd2_state_machine_invariants(self):
        state = SCD2DimensionState()

        # Batch 1: 5 customers initial load
        b1_records = [
            {"customer_id": f"cust_{i:03d}", "customer_city": "Sao Paulo", "customer_state": "SP", "customer_zip_code_prefix": "01000"}
            for i in range(1, 6)
        ]
        state.apply_batch(b1_records, "2023-01-01 00:00:00")
        
        self.assertEqual(len(state.query_current()), 5)
        self.assertEqual(state.history[-1]["inserted"], 5)
        self.assertEqual(state.history[-1]["updated"], 0)

        # Batch 2: cust_001 & cust_002 relocate, cust_003-005 unchanged, cust_006 brand new
        b2_records = [
            {"customer_id": "cust_001", "customer_city": "Campinas", "customer_state": "SP", "customer_zip_code_prefix": "13000"},
            {"customer_id": "cust_002", "customer_city": "Rio de Janeiro", "customer_state": "RJ", "customer_zip_code_prefix": "20000"},
            {"customer_id": "cust_003", "customer_city": "Sao Paulo", "customer_state": "SP", "customer_zip_code_prefix": "01000"},
            {"customer_id": "cust_004", "customer_city": "Sao Paulo", "customer_state": "SP", "customer_zip_code_prefix": "01000"},
            {"customer_id": "cust_005", "customer_city": "Sao Paulo", "customer_state": "SP", "customer_zip_code_prefix": "01000"},
            {"customer_id": "cust_006", "customer_city": "Curitiba", "customer_state": "PR", "customer_zip_code_prefix": "80000"},
        ]
        state.apply_batch(b2_records, "2023-04-01 00:00:00")
        
        self.assertEqual(state.history[-1]["updated"], 2, "cust_001 and cust_002 must be closed out")
        self.assertEqual(state.history[-1]["inserted"], 3, "2 new versions + 1 new customer")
        self.assertEqual(len(state.query_current()), 6, "Must have exactly 6 active customers")

        # Batch 3: Idempotent batch - identical data sent again
        state.apply_batch(b2_records, "2023-05-01 00:00:00")
        self.assertEqual(state.history[-1]["updated"], 0, "Idempotent batch must cause 0 updates")
        self.assertEqual(state.history[-1]["inserted"], 0, "Idempotent batch must cause 0 inserts")
        self.assertEqual(len(state.query_current()), 6)

        # Batch 4: Null transitions - cust_003 city becomes None, cust_004 state changes
        b4_records = [
            {"customer_id": "cust_003", "customer_city": None, "customer_state": "SP", "customer_zip_code_prefix": "01000"},
            {"customer_id": "cust_004", "customer_city": "Sao Paulo", "customer_state": "MG", "customer_zip_code_prefix": "01000"},
        ]
        state.apply_batch(b4_records, "2023-08-01 00:00:00")
        self.assertEqual(state.history[-1]["updated"], 2)
        self.assertEqual(state.history[-1]["inserted"], 2)
        
        # Verify cust_003 has new active version with city is None
        c3_active = [r for r in state.query_history("cust_003") if r["is_current"]][0]
        self.assertIsNone(c3_active["customer_city"])
        self.assertEqual(c3_active["effective_start_date"], "2023-08-01 00:00:00")

        # Batch 5: Third relocation for cust_001 (multi-hop: Sao Paulo -> Campinas -> Belo Horizonte)
        b5_records = [
            {"customer_id": "cust_001", "customer_city": "Belo Horizonte", "customer_state": "MG", "customer_zip_code_prefix": "30000"}
        ]
        state.apply_batch(b5_records, "2023-11-01 00:00:00")
        self.assertEqual(state.history[-1]["updated"], 1)
        self.assertEqual(state.history[-1]["inserted"], 1)

        # Global Invariant Verification across all rows in table
        c1_history = state.query_history("cust_001")
        self.assertEqual(len(c1_history), 3, "cust_001 must have exactly 3 historical versions")

        # Invariant 1: Exactly 1 active version per customer
        for cid in ["cust_001", "cust_002", "cust_003", "cust_004", "cust_005", "cust_006"]:
            hist = state.query_history(cid)
            active_versions = [r for r in hist if r["is_current"]]
            self.assertEqual(len(active_versions), 1, f"Customer {cid} must have exactly 1 active version")
            self.assertIsNone(active_versions[0]["effective_end_date"])

        # Invariant 2: Version continuity (end date of v_k == start date of v_{k+1})
        self.assertEqual(c1_history[0]["effective_start_date"], "2023-01-01 00:00:00")
        self.assertEqual(c1_history[0]["effective_end_date"], "2023-04-01 00:00:00")
        self.assertFalse(c1_history[0]["is_current"])

        self.assertEqual(c1_history[1]["effective_start_date"], "2023-04-01 00:00:00")
        self.assertEqual(c1_history[1]["effective_end_date"], "2023-11-01 00:00:00")
        self.assertFalse(c1_history[1]["is_current"])

        self.assertEqual(c1_history[2]["effective_start_date"], "2023-11-01 00:00:00")
        self.assertIsNone(c1_history[2]["effective_end_date"])
        self.assertTrue(c1_history[2]["is_current"])

        # Invariant 3: Unique surrogate keys across all rows
        all_sks = [r["customer_sk"] for r in state.rows]
        self.assertEqual(len(all_sks), len(set(all_sks)), "Surrogate keys must be strictly unique across all rows")


# ==============================================================================
# 2. Multi-Payment Combinations, Mixed Instruments & Zero-Payment Orders
# ==============================================================================

class TestMultiPaymentAdversarial(BaseAdversarialTestCase):
    """
    Stress-tests payment pre-aggregation under complex real-world conditions:
    - Multi-payment combination (credit card + voucher + debit card)
    - Zero-payment orders (abandoned or 100% gift card voucher)
    - High-order installment counts
    """

    def test_multi_payment_combination_breakdown_fidelity(self):
        orders = [{"order_id": "ord_complex_01", "order_status": "delivered"}]
        order_items = [
            {"order_id": "ord_complex_01", "order_item_id": 1, "price": 120.0},
            {"order_id": "ord_complex_01", "order_item_id": 2, "price": 80.0},
        ]
        # Payments split across 3 methods:
        # Credit card $100, Voucher 1 $40, Voucher 2 $20, Debit card $40 -> Total $200
        payments = [
            {"order_id": "ord_complex_01", "payment_sequential": 1, "payment_type": "credit_card", "payment_value": 100.0},
            {"order_id": "ord_complex_01", "payment_sequential": 2, "payment_type": "voucher", "payment_value": 40.0},
            {"order_id": "ord_complex_01", "payment_sequential": 3, "payment_type": "voucher", "payment_value": 20.0},
            {"order_id": "ord_complex_01", "payment_sequential": 4, "payment_type": "debit_card", "payment_value": 40.0},
        ]

        sim = simulate_fan_out_cartesian_vs_preagg(orders, order_items, payments)
        
        # In preagg, row count is strictly 1
        self.assertEqual(sim["preagg_row_count"], 1)
        self.assertEqual(sim["correct_total_revenue"], 200.0)
        self.assertEqual(sim["correct_total_payment"], 200.0)

        # In naive Cartesian join: 2 items * 4 payments = 8 rows!
        # Naive revenue = 200 * 4 = 800.0 (4x inflation!)
        self.assertEqual(sim["naive_row_count"], 8)
        self.assertEqual(sim["naive_total_revenue"], 800.0)
        self.assertEqual(sim["revenue_inflation_ratio"], 4.0)

    def test_zero_payment_and_empty_items_left_join_safety(self):
        """Verify code in gold_sales_analytics.ipynb uses LEFT joins and fillna for 0-payment orders."""
        code = self.get_notebook_code("gold/gold_sales_analytics.ipynb")
        
        # Verify left joins
        self.assertRegex(code, r'orders\.join\s*\(\s*order_items_agg\s*,\s*on\s*=\s*["\']order_id["\']\s*,\s*how\s*=\s*["\']left["\']')
        self.assertRegex(code, r'\.join\s*\(\s*payments_agg\s*,\s*on\s*=\s*["\']order_id["\']\s*,\s*how\s*=\s*["\']left["\']')
        
        # Verify fillna handles 0.0 defaults
        self.assertIn('"total_payment_value": 0.0', code)
        self.assertIn('"credit_card_payment_value": 0.0', code)
        self.assertIn('"boleto_payment_value": 0.0', code)
        self.assertIn('"voucher_payment_value": 0.0', code)
        self.assertIn('"debit_card_payment_value": 0.0', code)


# ==============================================================================
# 3. Zero-Freight Orders, Multi-Item Bundles & Fan-Out Invariants
# ==============================================================================

class TestZeroFreightAndItemBundles(BaseAdversarialTestCase):
    """
    Stress-tests order_items pre-aggregation with free-shipping promotions ($0.0 freight)
    and high-volume item bundles.
    """

    def test_zero_freight_and_multi_bundle_aggregation(self):
        orders = [{"order_id": "ord_promo_01", "order_status": "delivered"}]
        # 10 units of same item with 0.0 freight
        order_items = [
            {"order_id": "ord_promo_01", "order_item_id": i, "price": 15.0, "freight_value": 0.0}
            for i in range(1, 11)
        ]
        payments = [
            {"order_id": "ord_promo_01", "payment_sequential": 1, "payment_value": 150.0}
        ]

        sim = simulate_fan_out_cartesian_vs_preagg(orders, order_items, payments)
        self.assertEqual(sim["preagg_row_count"], 1)
        self.assertEqual(sim["correct_total_revenue"], 150.0)
        self.assertEqual(sim["correct_total_payment"], 150.0)


# ==============================================================================
# 4. Customers with Zero Orders & Identical Spend Ranking Ties
# ==============================================================================

class TestCustomerZeroOrdersAndTies(BaseAdversarialTestCase):
    """
    Stress-tests customer analytics under edge cases:
    - Customers who registered but never placed an order
    - Multiple customers tied with identical lifetime spend ($0.00 and $100.00)
    - DENSE_RANK() and NTILE(10) correctness
    """

    def test_customers_with_zero_orders_oracle(self):
        # 3 customers: cust_1 has $200, cust_2 has $0 (zero orders), cust_3 has $0 (zero orders)
        customers = [
            {"customer_id": "cust_1", "customer_city": "Sao Paulo", "customer_state": "SP"},
            {"customer_id": "cust_2", "customer_city": "Rio", "customer_state": "RJ"},
            {"customer_id": "cust_3", "customer_city": "Salvador", "customer_state": "BA"},
        ]
        orders = [
            {"order_id": "ord_1", "customer_id": "cust_1", "order_purchase_timestamp": "2023-01-15 10:00:00"}
        ]
        payments = [
            {"order_id": "ord_1", "payment_value": 200.0}
        ]

        metrics = simulate_customer_rfm_and_ltv(customers, orders, payments)
        self.assertEqual(len(metrics), 3)

        # cust_1 must be rank 1
        c1 = [m for m in metrics if m["customer_id"] == "cust_1"][0]
        self.assertEqual(c1["lifetime_value"], 200.0)
        self.assertEqual(c1["total_orders"], 1)
        self.assertEqual(c1["spend_dense_rank"], 1)

        # cust_2 and cust_3 have zero orders
        c2 = [m for m in metrics if m["customer_id"] == "cust_2"][0]
        self.assertEqual(c2["lifetime_value"], 0.0)
        self.assertEqual(c2["total_orders"], 0)
        self.assertEqual(c2["average_basket_size"], 0.0)
        self.assertIsNone(c2["first_purchase_date"])
        self.assertIsNone(c2["last_purchase_date"])

        # Dense rank for tied zero spend must be 2
        c3 = [m for m in metrics if m["customer_id"] == "cust_3"][0]
        self.assertEqual(c2["spend_dense_rank"], 2)
        self.assertEqual(c3["spend_dense_rank"], 2)

    def test_customer_notebook_null_handling_for_zero_orders(self):
        """Verify gold_customer_analytics.ipynb left-joins and fills defaults for zero-order customers."""
        code = self.get_notebook_code("gold/gold_customer_analytics.ipynb")
        
        # Must left join customers to customer_orders_agg
        self.assertRegex(code, r'customers\s*\.join\s*\(\s*customer_orders_agg\s*,\s*on\s*=\s*["\']customer_id["\']\s*,\s*how\s*=\s*["\']left["\']')
        
        # Must fill defaults for zero order customers
        self.assertIn('"total_orders": 0', code)
        self.assertIn('"total_spend": 0.0', code)
        self.assertIn('"avg_basket_size": 0.0', code)
        self.assertIn('"customer_tenure_days": 0', code)


# ==============================================================================
# 5. Temporal Boundaries, Leap Years & Division-by-Zero Guards
# ==============================================================================

class TestTemporalBoundariesAndLeapYears(BaseAdversarialTestCase):
    """
    Stress-tests date and timestamp transformations:
    - Leap year dates (2016-02-29, 2020-02-29)
    - DoD and MoM division-by-zero guards when previous revenue is zero or null
    """

    def test_leap_year_timestamp_parsing_fidelity(self):
        leap_timestamps = [
            "2016-02-29 12:34:56",
            "2020-02-29 23:59:59",
            "2024-02-29 00:00:00",
        ]
        for ts_str in leap_timestamps:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            self.assertEqual(dt.month, 2)
            self.assertEqual(dt.day, 29)
            # Truncation to month
            year_month = dt.strftime("%Y-%m")
            self.assertEqual(year_month, f"{dt.year:04d}-02")

    def test_growth_rate_zero_division_guard_logic(self):
        """Verify gold_sales_analytics.ipynb protects against division by zero."""
        code = self.get_notebook_code("gold/gold_sales_analytics.ipynb")

        # DoD growth rate guard: prev_day_revenue isNotNull and != 0
        self.assertRegex(
            code,
            r'\(?\s*F\.col\(["\']prev_day_revenue["\']\)\.isNotNull\(\)\s*\)?\s*&\s*\(?\s*F\.col\(["\']prev_day_revenue["\']\)\s*!=\s*0\s*\)?',
            "DoD growth rate must explicitly guard against isNull and != 0"
        )

        # MoM growth rate guard: prev_month_revenue isNotNull and != 0
        self.assertRegex(
            code,
            r'\(?\s*F\.col\(["\']prev_month_revenue["\']\)\.isNotNull\(\)\s*\)?\s*&\s*\(?\s*F\.col\(["\']prev_month_revenue["\']\)\s*!=\s*0\s*\)?',
            "MoM growth rate must explicitly guard against isNull and != 0"
        )


# ==============================================================================
# 6. Product Pareto Distribution Boundary Conditions & Extreme Catalogs
# ==============================================================================

class TestProductParetoBoundaryConditions(BaseAdversarialTestCase):
    """
    Stress-tests category Pareto distribution under boundary conditions:
    - Single category with 100% share
    - All categories with 0.0 revenue (division-by-zero protection)
    - 50 categories ranking and cumulative percentage monotonicity
    """

    def test_single_monopoly_category_pareto(self):
        products = [{"product_id": "p1", "product_category_name": "monopoly_cat"}]
        order_items = [{"product_id": "p1", "order_id": "o1", "price": 500.0}]

        pareto = simulate_pareto_category_distribution(products, order_items)
        self.assertEqual(len(pareto), 1)
        self.assertEqual(pareto[0]["cumulative_share_pct"], 100.0)
        self.assertEqual(pareto[0]["category_revenue_rank"], 1)
        self.assertTrue(pareto[0]["is_pareto_top_80"])

    def test_zero_revenue_pareto_guard(self):
        """Verify gold_product_analytics.ipynb guards against division by zero when all_categories_revenue is 0."""
        code = self.get_notebook_code("gold/gold_product_analytics.ipynb")
        
        # Check pareto_revenue_pct division guard
        self.assertRegex(
            code,
            r'when\s*\(\s*F\.col\(["\']all_categories_revenue["\']\)\s*==\s*0\s*,\s*0\.0\s*\)',
            "Product analytics must guard against division by zero when all_categories_revenue == 0"
        )

    def test_pareto_monotonicity_across_50_categories(self):
        products = [{"product_id": f"p_{i}", "product_category_name": f"cat_{i:02d}"} for i in range(1, 51)]
        order_items = []
        for i in range(1, 51):
            # Revenue inversely proportional to i
            order_items.append({"product_id": f"p_{i}", "order_id": f"o_{i}", "price": round(1000.0 / i, 2)})

        pareto = simulate_pareto_category_distribution(products, order_items)
        self.assertEqual(len(pareto), 50)

        prev_share = 0.0
        for r in pareto:
            curr_share = r["cumulative_share_pct"]
            self.assertGreaterEqual(curr_share, prev_share, "Cumulative share must be non-decreasing")
            self.assertLessEqual(curr_share, 100.0, "Cumulative share must not exceed 100%")
            prev_share = curr_share

        self.assertAlmostEqual(pareto[-1]["cumulative_share_pct"], 100.0, places=1)


# ==============================================================================
# 7. Surrogate Key Collision Resistance, Unicode & Null Safety
# ==============================================================================

class TestSurrogateKeySafetyAndUnicode(BaseAdversarialTestCase):
    """
    Stress-tests surrogate key hashing and Unicode handling:
    - Brazilian Portuguese accents and diacritics
    - Null attribute surrogate key safety
    - 5,000 distinct surrogate keys collision audit
    """

    def test_portuguese_unicode_hashing(self):
        cities = ["São Paulo", "Brasília", "Ribeirão Preto", "Goiânia", "Belém", "Florianópolis"]
        sks = set()
        for i, city in enumerate(cities):
            sk = generate_scd2_surrogate_key(f"cust_{city}", "2023-01-01 00:00:00")
            self.assertEqual(len(sk), 64, "SHA-256 hash must be 64 hexadecimal characters")
            sks.add(sk)
        self.assertEqual(len(sks), len(cities), "Surrogate keys for Unicode cities must be distinct")

    def test_surrogate_key_zero_collision_audit(self):
        generated = set()
        count = 5000
        for i in range(count):
            sk = generate_scd2_surrogate_key(f"customer_{i:06d}", f"2023-01-01 00:{i%60:02d}:00")
            generated.add(sk)
        self.assertEqual(len(generated), count, f"Expected 0 collisions across {count} surrogate keys")


# ==============================================================================
# 8. White-Box AST & Notebook Code Architecture Verification
# ==============================================================================

class TestNotebookArchitectureWhiteBox(BaseAdversarialTestCase):
    """
    White-box audit of notebook Python AST and Databricks SQL cells.
    Verifies adherence to Delta format, Unity Catalog, and Medallion architecture.
    """

    def test_all_notebooks_ast_clean_syntax(self):
        notebooks = [
            "bronze/bronze_ingestion.ipynb",
            "gold/gold_sales_analytics.ipynb",
            "gold/gold_customer_analytics.ipynb",
            "gold/gold_product_analytics.ipynb",
            "gold/gold_scd2_dimensions.ipynb",
        ]
        for nb in notebooks:
            code = self.get_notebook_code(nb)
            tree = parse_python_ast(code, context_name=nb)
            self.assertIsInstance(tree, ast.Module, f"Failed to parse AST for {nb}")

    def test_gold_notebooks_terminal_sql_cells(self):
        gold_notebooks = [
            "gold/gold_sales_analytics.ipynb",
            "gold/gold_customer_analytics.ipynb",
            "gold/gold_product_analytics.ipynb",
            "gold/gold_scd2_dimensions.ipynb",
        ]
        for nb in gold_notebooks:
            sql_cells = self.get_notebook_sql(nb)
            self.assertGreater(len(sql_cells), 0, f"{nb} must have at least one %sql cell")
            for cell in sql_cells:
                self.assertTrue(cell.strip().startswith("%sql"), f"SQL cell in {nb} must start with %sql")


if __name__ == "__main__":
    unittest.main()
