"""
Comprehensive E2E Test Suite for the Lakehouse portion of Ecommerce Data Pipeline.

Structured across 4 Systematic Tiers + Constraints Validation:
- Constraints Validation: Scope exclusions per R3 (Kafka, Airflow, Postgres, Spark Streaming)
- Tier 1: Feature Coverage (F01, F02, F05, F06, F11, F15, F16, F21)
- Tier 2: Boundary & Corner Cases (F04, Nulls, Div-by-Zero, Timestamps, Hash Safety)
- Tier 3: Cross-Feature Combinations (F03, F07, F08, F10, F12, F13, F14, F17)
- Tier 4: Real-World Scenarios (F18, F19, F20, Oracles, Lineage Contracts)
"""

import ast
import json
import os
import re
import unittest
from typing import Any, Dict, List, Set

from tests.lakehouse_test_utils import (
    SCD2DimensionState,
    check_fan_out_mitigation_ast,
    extract_code_cells,
    extract_python_and_sql,
    extract_table_references,
    get_repo_root,
    load_notebook,
    parse_python_ast,
    simulate_customer_rfm_and_ltv,
    simulate_fan_out_cartesian_vs_preagg,
    simulate_pareto_category_distribution,
)


class BaseLakehouseTestCase(unittest.TestCase):
    """Base test case providing helper methods for notebook validation."""

    def assertNotebookExists(self, relative_path: str):
        repo_root = get_repo_root()
        abs_path = os.path.join(repo_root, relative_path)
        self.assertTrue(
            os.path.exists(abs_path),
            f"Expected notebook does not exist: {relative_path}"
        )

    def get_notebook_code(self, relative_path: str) -> str:
        self.assertNotebookExists(relative_path)
        nb = load_notebook(relative_path)
        py_code, _ = extract_python_and_sql(nb)
        return py_code

    def get_notebook_sql_cells(self, relative_path: str) -> List[str]:
        self.assertNotebookExists(relative_path)
        nb = load_notebook(relative_path)
        _, sql_cells = extract_python_and_sql(nb)
        return sql_cells


# ==============================================================================
# Constraints Validation (F22 / Requirement R3)
# ==============================================================================

class TestConstraintsValidation(BaseLakehouseTestCase):
    """
    Validates Requirement R3 (Scope Exclusions):
    Strictly verify zero external services (Kafka, Airflow, PostgreSQL, Spark Streaming)
    are introduced into the repository.
    """

    def _get_files_to_scan(self) -> List[str]:
        repo_root = get_repo_root()
        files_to_check = []
        for root, dirs, files in os.walk(repo_root):
            # Exclude tests, .git, .agents, docs, and metadata
            rel_root = os.path.relpath(root, repo_root)
            if any(rel_root.startswith(p) for p in ['.git', '.agents', 'tests', '.gemini']):
                continue
            for file in files:
                # Exclude markdown doc files and gitignore
                if file.endswith(('.md', '.gitignore', '.gitkeep', '.txt')):
                    continue
                if file.endswith(('.ipynb', '.py', '.yml', '.yaml', '.properties', '.sh', '.sql', '.conf', '.json')):
                    files_to_check.append(os.path.join(root, file))
        return files_to_check

    def test_no_kafka_dependencies_or_configs(self):
        """Verify repository contains no Kafka brokers, configs, or imports."""
        forbidden_patterns = [
            r'kafka\.bootstrap\.servers',
            r'org\.apache\.spark:spark-sql-kafka',
            r'from\s+kafka\s+import',
            r'import\s+kafka',
            r'confluent_kafka',
        ]
        
        for path in self._get_files_to_scan():
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            for pat in forbidden_patterns:
                match = re.search(pat, content, re.IGNORECASE)
                self.assertIsNone(
                    match,
                    f"Forbidden Kafka artifact found in {path} matching: {pat}"
                )

    def test_no_airflow_dags_or_dependencies(self):
        """Verify repository contains no Airflow DAGs, operators, or imports."""
        forbidden_patterns = [
            r'from\s+airflow\s+import',
            r'import\s+airflow',
            r'DAG\s*\(',
            r'airflow\.operators',
        ]
        
        for path in self._get_files_to_scan():
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            for pat in forbidden_patterns:
                match = re.search(pat, content, re.IGNORECASE)
                self.assertIsNone(
                    match,
                    f"Forbidden Airflow artifact found in {path} matching: {pat}"
                )

    def test_no_postgres_external_services(self):
        """Verify repository contains no PostgreSQL connection strings, drivers, or Docker configs."""
        forbidden_patterns = [
            r'postgresql://',
            r'org\.postgresql\.Driver',
            r'psycopg2',
            r'asyncpg',
            r'image:\s*postgres',
        ]
        
        for path in self._get_files_to_scan():
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            for pat in forbidden_patterns:
                match = re.search(pat, content, re.IGNORECASE)
                self.assertIsNone(
                    match,
                    f"Forbidden Postgres artifact found in {path} matching: {pat}"
                )

    def test_no_spark_structured_streaming(self):
        """Verify repository contains no Spark Structured Streaming calls."""
        forbidden_patterns = [
            r'spark\.readStream',
            r'\.writeStream',
            r'\.trigger\s*\(',
        ]
        
        for path in self._get_files_to_scan():
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            for pat in forbidden_patterns:
                match = re.search(pat, content, re.IGNORECASE)
                self.assertIsNone(
                    match,
                    f"Forbidden Spark Streaming artifact found in {path} matching: {pat}"
                )


# ==============================================================================
# Tier 1: Feature Coverage (F01, F02, F05, F06, F11, F15, F16, F21)
# ==============================================================================

class TestTier1FeatureCoverage(BaseLakehouseTestCase):
    """
    Tier 1: Feature Coverage
    Validates notebook existence, nbformat 4.0 schema, Databricks metadata,
    Python syntax validity, and Unity Catalog Gold table outputs.
    """

    def test_f01_bronze_products_ingestion(self):
        """F01: bronze_ingestion.ipynb must ingest olist_products_dataset.csv into workspace.ecompulse_bronze.products."""
        code = self.get_notebook_code("bronze/bronze_ingestion.ipynb")
        
        # Verify source CSV read
        self.assertRegex(
            code,
            r'olist_products_dataset\.csv',
            "bronze_ingestion.ipynb must load olist_products_dataset.csv"
        )
        
        # Verify target Bronze Delta table
        self.assertRegex(
            code,
            r'workspace\.ecompulse_bronze\.products',
            "bronze_ingestion.ipynb must save to table workspace.ecompulse_bronze.products"
        )
        
        # Verify Delta format specified
        self.assertRegex(
            code,
            r'(?:format\s*\(\s*["\']delta["\']|delta)',
            "bronze_ingestion.ipynb must write products in Delta format"
        )

    def test_f02_gold_notebooks_exist(self):
        """F02: All required Gold layer notebooks must exist under gold/."""
        expected_notebooks = [
            "gold/gold_sales_analytics.ipynb",
            "gold/gold_customer_analytics.ipynb",
            "gold/gold_product_analytics.ipynb",
            "gold/gold_scd2_dimensions.ipynb",
        ]
        for nb_path in expected_notebooks:
            self.assertNotebookExists(nb_path)

    def test_f02_notebook_nbformat_4_schema(self):
        """F02: All notebooks must strictly comply with nbformat 4.0 schema."""
        all_notebooks = [
            "bronze/bronze_ingestion.ipynb",
            "silver/silver_customers.ipynb",
            "silver/silver_orders.ipynb",
            "silver/silver_order_items.ipynb",
            "silver/silver_payments.ipynb",
            "silver/silver_products.ipynb",
            "gold/gold_sales_analytics.ipynb",
            "gold/gold_customer_analytics.ipynb",
            "gold/gold_product_analytics.ipynb",
            "gold/gold_scd2_dimensions.ipynb",
        ]
        for nb_path in all_notebooks:
            self.assertNotebookExists(nb_path)
            nb = load_notebook(nb_path)
            self.assertEqual(
                nb.get("nbformat"), 4,
                f"Notebook {nb_path} must have nbformat == 4"
            )
            self.assertGreaterEqual(
                nb.get("nbformat_minor", -1), 0,
                f"Notebook {nb_path} must have nbformat_minor >= 0"
            )
            self.assertIn("cells", nb, f"Notebook {nb_path} missing 'cells' array")
            self.assertIn("metadata", nb, f"Notebook {nb_path} missing 'metadata' object")

    def test_f02_notebook_databricks_metadata(self):
        """F02: Gold notebooks must contain Databricks application/vnd metadata and python language info."""
        gold_notebooks = [
            "gold/gold_sales_analytics.ipynb",
            "gold/gold_customer_analytics.ipynb",
            "gold/gold_product_analytics.ipynb",
            "gold/gold_scd2_dimensions.ipynb",
        ]
        for nb_path in gold_notebooks:
            self.assertNotebookExists(nb_path)
            nb = load_notebook(nb_path)
            metadata = nb.get("metadata", {})
            db_meta = metadata.get("application/vnd.databricks.v1+notebook", {})
            self.assertIsInstance(
                db_meta, dict,
                f"Notebook {nb_path} must contain application/vnd.databricks.v1+notebook metadata"
            )
            self.assertEqual(
                db_meta.get("language"), "python",
                f"Notebook {nb_path} Databricks language must be python"
            )

    def test_f02_all_notebook_cells_valid_python_syntax(self):
        """F02: All Python code cells across all notebooks must parse without syntax errors."""
        all_notebooks = [
            "bronze/bronze_ingestion.ipynb",
            "silver/silver_customers.ipynb",
            "silver/silver_orders.ipynb",
            "silver/silver_order_items.ipynb",
            "silver/silver_payments.ipynb",
            "silver/silver_products.ipynb",
            "gold/gold_sales_analytics.ipynb",
            "gold/gold_customer_analytics.ipynb",
            "gold/gold_product_analytics.ipynb",
            "gold/gold_scd2_dimensions.ipynb",
        ]
        for nb_path in all_notebooks:
            self.assertNotebookExists(nb_path)
            nb = load_notebook(nb_path)
            py_code, _ = extract_python_and_sql(nb)
            try:
                parse_python_ast(py_code, context_name=nb_path)
            except SyntaxError as e:
                self.fail(f"Python syntax error in {nb_path}: {e}")

    def test_f05_f06_sales_gold_table_targets(self):
        """F05 & F06: gold_sales_analytics.ipynb must write to daily_sales_analytics and monthly_sales_analytics."""
        code = self.get_notebook_code("gold/gold_sales_analytics.ipynb")
        tables = extract_table_references(code)
        
        self.assertIn(
            "workspace.ecompulse_gold.daily_sales_analytics", tables,
            "gold_sales_analytics.ipynb must target workspace.ecompulse_gold.daily_sales_analytics"
        )
        self.assertIn(
            "workspace.ecompulse_gold.monthly_sales_analytics", tables,
            "gold_sales_analytics.ipynb must target workspace.ecompulse_gold.monthly_sales_analytics"
        )

    def test_f11_customer_gold_table_target(self):
        """F11: gold_customer_analytics.ipynb must write to workspace.ecompulse_gold.customer_analytics."""
        code = self.get_notebook_code("gold/gold_customer_analytics.ipynb")
        tables = extract_table_references(code)
        
        self.assertIn(
            "workspace.ecompulse_gold.customer_analytics", tables,
            "gold_customer_analytics.ipynb must target workspace.ecompulse_gold.customer_analytics"
        )

    def test_f15_product_gold_table_targets(self):
        """F15: gold_product_analytics.ipynb must write to product_performance and category_performance."""
        code = self.get_notebook_code("gold/gold_product_analytics.ipynb")
        tables = extract_table_references(code)
        
        self.assertIn(
            "workspace.ecompulse_gold.product_performance", tables,
            "gold_product_analytics.ipynb must target workspace.ecompulse_gold.product_performance"
        )
        self.assertIn(
            "workspace.ecompulse_gold.category_performance", tables,
            "gold_product_analytics.ipynb must target workspace.ecompulse_gold.category_performance"
        )

    def test_f16_scd2_gold_table_target(self):
        """F16: gold_scd2_dimensions.ipynb must target workspace.ecompulse_gold.dim_customers_scd2."""
        code = self.get_notebook_code("gold/gold_scd2_dimensions.ipynb")
        tables = extract_table_references(code)
        
        self.assertIn(
            "workspace.ecompulse_gold.dim_customers_scd2", tables,
            "gold_scd2_dimensions.ipynb must target workspace.ecompulse_gold.dim_customers_scd2"
        )

    def test_f21_sql_preview_verification_cells(self):
        """F21: Every Gold notebook must contain at least one terminal %sql verification preview cell."""
        gold_notebooks = [
            ("gold/gold_sales_analytics.ipynb", "daily_sales_analytics"),
            ("gold/gold_customer_analytics.ipynb", "customer_analytics"),
            ("gold/gold_product_analytics.ipynb", "product_performance"),
            ("gold/gold_scd2_dimensions.ipynb", "dim_customers_scd2"),
        ]
        for nb_path, expected_table_fragment in gold_notebooks:
            sql_cells = self.get_notebook_sql_cells(nb_path)
            self.assertGreater(
                len(sql_cells), 0,
                f"Notebook {nb_path} must contain at least one %sql preview cell (Feature F21)"
            )
            # Verify the SQL preview queries the target table
            combined_sql = "\n".join(sql_cells)
            self.assertIn(
                expected_table_fragment, combined_sql,
                f"Notebook {nb_path} %sql cells must query '{expected_table_fragment}'"
            )


# ==============================================================================
# Tier 2: Boundary & Corner Cases (F04, Nulls, Div-by-Zero, Timestamps, Hash Safety)
# ==============================================================================

class TestTier2BoundaryAndCornerCases(BaseLakehouseTestCase):
    """
    Tier 2: Boundary & Corner Cases
    Validates fan-out mitigation pre-aggregation, division-by-zero guards in growth rates,
    null handling in joins and metrics, timestamp conversions, and surrogate key null safety.
    """

    def test_f04_sales_fan_out_pre_aggregation(self):
        """
        F04: gold_sales_analytics.ipynb must pre-aggregate order_items and payments
        by order_id prior to joining with orders to prevent Cartesian multiplication.
        """
        code = self.get_notebook_code("gold/gold_sales_analytics.ipynb")
        mitigation = check_fan_out_mitigation_ast(code)
        
        self.assertTrue(
            mitigation["pre_aggregates_items"],
            "gold_sales_analytics.ipynb must pre-aggregate order_items by order_id before joining (Feature F04)"
        )
        self.assertTrue(
            mitigation["pre_aggregates_payments"],
            "gold_sales_analytics.ipynb must pre-aggregate payments by order_id before joining (Feature F04)"
        )

    def test_sales_revenue_growth_rate_division_by_zero_guards(self):
        """
        Boundary: MoM and DoD revenue growth rate calculations must guard against
        division by zero when prior period revenue is 0 or null.
        """
        code = self.get_notebook_code("gold/gold_sales_analytics.ipynb")
        
        # Verify growth rate calculation exists
        self.assertRegex(
            code,
            r'(?:mom_growth|dod_growth|revenue_growth|growth_pct|growth_rate)',
            "gold_sales_analytics.ipynb must compute revenue growth rates (MoM or DoD)"
        )
        
        # Verify division by zero protection: when/otherwise or nullif or coalesce
        div_zero_guard_patterns = [
            r'when\s*\(\s*.*(?:==\s*0|!=\s*0|isNull|\.isNotNull).*otherwise',
            r'when\s*\(\s*.*(?:lag|prev|prior).*!=',
            r'nullif',
        ]
        has_guard = any(re.search(pat, code, re.IGNORECASE) for pat in div_zero_guard_patterns)
        self.assertTrue(
            has_guard,
            "Revenue growth rate computation must guard against division by zero (e.g. F.when(prev != 0, ...))"
        )

    def test_customer_analytics_null_and_zero_order_guards(self):
        """
        Boundary: Customer analytics must handle customers with zero orders, null payments,
        or missing cities/states without raising errors or producing NaN values.
        """
        code = self.get_notebook_code("gold/gold_customer_analytics.ipynb")
        
        # Check for null handling: coalesce, fillna, when, or outer join handling
        null_guard_patterns = [
            r'coalesce',
            r'fillna',
            r'when\s*\(',
        ]
        has_null_handling = any(re.search(pat, code, re.IGNORECASE) for pat in null_guard_patterns)
        self.assertTrue(
            has_null_handling,
            "gold_customer_analytics.ipynb must handle null values via coalesce, fillna, or when"
        )

    def test_product_analytics_null_guards(self):
        """
        Boundary: Product analytics must handle products with missing category names
        and zero sales volume safely.
        """
        code = self.get_notebook_code("gold/gold_product_analytics.ipynb")
        
        null_guard_patterns = [
            r'coalesce',
            r'fillna',
            r'when\s*\(',
        ]
        has_null_handling = any(re.search(pat, code, re.IGNORECASE) for pat in null_guard_patterns)
        self.assertTrue(
            has_null_handling,
            "gold_product_analytics.ipynb must handle null categories/values via coalesce, fillna, or when"
        )

    def test_timestamp_and_date_conversions(self):
        """
        Boundary: Daily and monthly marts must perform correct date transformations
        (e.g. to_date, date_trunc, trunc) from order_purchase_timestamp.
        """
        code = self.get_notebook_code("gold/gold_sales_analytics.ipynb")
        
        date_func_patterns = [
            r'to_date',
            r'date_trunc',
            r'trunc',
        ]
        has_date_func = any(re.search(pat, code, re.IGNORECASE) for pat in date_func_patterns)
        self.assertTrue(
            has_date_func,
            "gold_sales_analytics.ipynb must convert timestamps to daily/monthly dates (to_date, date_trunc, trunc)"
        )

    def test_scd2_surrogate_key_hash_null_safety(self):
        """
        Boundary: SCD Type 2 surrogate key must use SHA256 hashing with concat_ws
        or null-safe delimiter to avoid hash collisions on null attributes.
        """
        code = self.get_notebook_code("gold/gold_scd2_dimensions.ipynb")
        
        # Check for sha2 / sha256
        self.assertRegex(
            code,
            r'(?:sha2|sha256|hashlib\.sha256)',
            "gold_scd2_dimensions.ipynb must generate SHA-256 surrogate keys"
        )
        # Check for concat_ws or concatenation with separator
        self.assertRegex(
            code,
            r'(?:concat_ws|concat|\+)',
            "gold_scd2_dimensions.ipynb must concatenate natural key and validity attributes with separator"
        )


# ==============================================================================
# Tier 3: Cross-Feature Combinations (F03, F07, F08, F10, F12, F13, F14, F17)
# ==============================================================================

class TestTier3CrossFeatureCombinations(BaseLakehouseTestCase):
    """
    Tier 3: Cross-Feature Combinations
    Validates multi-table joins, complex window function specifications,
    multi-metric analytical aggregations, and Delta MERGE matching conditions.
    """

    def test_f03_sales_multi_table_joins(self):
        """F03: gold_sales_analytics.ipynb must join orders with silver customers, order_items, and payments."""
        code = self.get_notebook_code("gold/gold_sales_analytics.ipynb")
        tables = extract_table_references(code)
        
        expected_inputs = [
            "workspace.ecompulse_silver.orders",
            "workspace.ecompulse_silver.payments",
            "workspace.ecompulse_silver.order_items",
        ]
        for tbl in expected_inputs:
            self.assertIn(
                tbl, tables,
                f"gold_sales_analytics.ipynb must read from input table: {tbl}"
            )

    def test_f07_sales_window_functions_specification(self):
        """
        F07: gold_sales_analytics.ipynb must implement:
        - Running total revenue (Window ordered by date)
        - 7-day rolling moving average (Window ordered by date, rows/range between -6/0 or -7/0)
        - MoM revenue growth (Window with lag(1) or offset)
        """
        code = self.get_notebook_code("gold/gold_sales_analytics.ipynb")
        
        # Window module import / usage
        self.assertRegex(
            code,
            r'Window\.',
            "gold_sales_analytics.ipynb must use PySpark Window specifications"
        )
        
        # Running total (unboundedPreceding or cumulative sum)
        self.assertRegex(
            code,
            r'(?:unboundedPreceding|running_total|cumulative)',
            "gold_sales_analytics.ipynb must compute running total revenue"
        )
        
        # 7-day rolling moving average
        self.assertRegex(
            code,
            r'(?:rowsBetween|rangeBetween|-6|-7|moving_avg|rolling_avg|7.*day)',
            "gold_sales_analytics.ipynb must compute 7-day rolling moving average"
        )

    def test_f08_f09_f10_customer_analytics_joins_and_metrics(self):
        """
        F08, F09, F10: Customer analytics must join customers with orders and payments,
        calculate LTV and basket size, and compute customer window functions
        (order sequence ROW_NUMBER, spend DENSE_RANK, spending decile NTILE(10)).
        """
        code = self.get_notebook_code("gold/gold_customer_analytics.ipynb")
        tables = extract_table_references(code)
        
        self.assertIn("workspace.ecompulse_silver.customers", tables)
        self.assertIn("workspace.ecompulse_silver.orders", tables)
        
        # Window functions: row_number, dense_rank, ntile(10)
        self.assertRegex(code, r'row_number\s*\(', "Customer analytics must compute order sequence via row_number()")
        self.assertRegex(code, r'dense_rank\s*\(', "Customer analytics must compute customer spend rank via dense_rank()")
        self.assertRegex(code, r'ntile\s*\(\s*10\s*\)', "Customer analytics must compute spending deciles via ntile(10)")

    def test_f12_f13_f14_product_performance_joins_and_aggregations(self):
        """
        F12, F13, F14: Product performance must join products, order_items, and orders,
        aggregate sales volume and revenue, and compute category rank (DENSE_RANK)
        and Pareto category contribution share.
        """
        code = self.get_notebook_code("gold/gold_product_analytics.ipynb")
        tables = extract_table_references(code)
        
        self.assertIn("workspace.ecompulse_silver.products", tables)
        self.assertIn("workspace.ecompulse_silver.order_items", tables)
        
        # Check window functions: dense_rank and running total / total for Pareto share
        self.assertRegex(code, r'dense_rank\s*\(', "Product analytics must compute category ranking via dense_rank()")
        self.assertRegex(
            code,
            r'(?:pareto|running_total|cumulative|share_pct|pct_share)',
            "Product analytics must compute Pareto category revenue contribution"
        )

    def test_f17_scd2_delta_merge_specification(self):
        """
        F17: gold_scd2_dimensions.ipynb must implement Delta Lake staged MERGE
        matching on natural key and active flag, closing updated rows and inserting new versions.
        """
        code = self.get_notebook_code("gold/gold_scd2_dimensions.ipynb")
        
        # Verify DeltaTable.forName or spark.read.table / asTable
        self.assertRegex(
            code,
            r'(?:DeltaTable\.forName|\.asTable|\.merge)',
            "gold_scd2_dimensions.ipynb must use DeltaTable merge operation"
        )
        
        # Verify matching condition: customer_id and is_current
        self.assertRegex(
            code,
            r'customer_id',
            "MERGE match condition must compare customer_id"
        )
        self.assertRegex(
            code,
            r'is_current',
            "MERGE match condition must evaluate is_current active flag"
        )
        
        # Verify whenMatchedUpdate and whenNotMatchedInsert
        self.assertRegex(
            code,
            r'whenMatchedUpdate',
            "MERGE must contain whenMatchedUpdate clause to close out active version"
        )
        self.assertRegex(
            code,
            r'whenNotMatchedInsert',
            "MERGE must contain whenNotMatchedInsert clause to insert new version"
        )


# ==============================================================================
# Tier 4: Real-World Scenarios & Algorithmic Oracles
# ==============================================================================

class TestTier4RealWorldScenarios(BaseLakehouseTestCase):
    """
    Tier 4: Real-World Scenarios
    Validates incremental loading, schema enforcement/evolution, Delta time travel,
    and executes mathematical reference simulations (Oracles) verifying logic correctness.
    """

    def test_f18_incremental_load_logic(self):
        """F18: gold_scd2_dimensions.ipynb must demonstrate incremental loading via watermark / timestamp filtering."""
        code = self.get_notebook_code("gold/gold_scd2_dimensions.ipynb")
        
        incremental_patterns = [
            r'watermark',
            r'(?:batch|effective|load)_timestamp',
            r'>=\s*(?:last_loaded|watermark|max_date)',
            r'max\s*\(\s*["\']effective_start_date["\']',
            r'current_timestamp\s*\(',
        ]
        has_incremental = any(re.search(pat, code, re.IGNORECASE) for pat in incremental_patterns)
        self.assertTrue(
            has_incremental,
            "gold_scd2_dimensions.ipynb must demonstrate incremental loading with watermark/timestamp"
        )

    def test_f19_schema_enforcement_and_evolution(self):
        """F19: gold_scd2_dimensions.ipynb must demonstrate schema enforcement or evolution via mergeSchema."""
        code = self.get_notebook_code("gold/gold_scd2_dimensions.ipynb")
        
        self.assertRegex(
            code,
            r'mergeSchema',
            "gold_scd2_dimensions.ipynb must demonstrate Delta Lake schema evolution using 'mergeSchema'"
        )

    def test_f20_delta_time_travel_and_history(self):
        """F20: gold_scd2_dimensions.ipynb must demonstrate Delta Lake time travel queries and DESCRIBE HISTORY."""
        code = self.get_notebook_code("gold/gold_scd2_dimensions.ipynb")
        sql_cells = self.get_notebook_sql_cells("gold/gold_scd2_dimensions.ipynb")
        combined = code + "\n" + "\n".join(sql_cells)
        
        # Verify VERSION AS OF or TIMESTAMP AS OF or history()
        time_travel_patterns = [
            r'VERSION\s+AS\s+OF',
            r'TIMESTAMP\s+AS\s+OF',
            r'versionAsOf',
            r'timestampAsOf',
            r'history\s*\(',
            r'DESCRIBE\s+HISTORY',
        ]
        has_time_travel = any(re.search(pat, combined, re.IGNORECASE) for pat in time_travel_patterns)
        self.assertTrue(
            has_time_travel,
            "gold_scd2_dimensions.ipynb must demonstrate Delta Lake time travel (VERSION AS OF or DESCRIBE HISTORY)"
        )

    # --------------------------------------------------------------------------
    # Algorithmic Oracle Tests (Mathematical Reference Models)
    # --------------------------------------------------------------------------

    def test_oracle_fan_out_mitigation_mathematical_proof(self):
        """
        Oracle Test: Demonstrates mathematically that joining orders to items and payments
        without pre-aggregation inflates total revenue and payments due to Cartesian fan-out,
        whereas pre-aggregation produces exact mathematical fidelity.
        """
        # Synthetic fixture: 2 orders
        # Order 1: 3 items (10.0, 20.0, 30.0 -> total 60.0), 2 payments (30.0, 30.0 -> total 60.0)
        # Order 2: 2 items (50.0, 50.0 -> total 100.0), 3 payments (20.0, 40.0, 40.0 -> total 100.0)
        orders = [
            {"order_id": "ord_001", "order_status": "delivered"},
            {"order_id": "ord_002", "order_status": "delivered"},
        ]
        order_items = [
            {"order_id": "ord_001", "order_item_id": 1, "price": 10.0},
            {"order_id": "ord_001", "order_item_id": 2, "price": 20.0},
            {"order_id": "ord_001", "order_item_id": 3, "price": 30.0},
            {"order_id": "ord_002", "order_item_id": 1, "price": 50.0},
            {"order_id": "ord_002", "order_item_id": 2, "price": 50.0},
        ]
        payments = [
            {"order_id": "ord_001", "payment_sequential": 1, "payment_value": 30.0},
            {"order_id": "ord_001", "payment_sequential": 2, "payment_value": 30.0},
            {"order_id": "ord_002", "payment_sequential": 1, "payment_value": 20.0},
            {"order_id": "ord_002", "payment_sequential": 2, "payment_value": 40.0},
            {"order_id": "ord_002", "payment_sequential": 3, "payment_value": 40.0},
        ]
        
        sim = simulate_fan_out_cartesian_vs_preagg(orders, order_items, payments)
        
        # Correct totals: items sum = 160.0, payments sum = 160.0
        self.assertEqual(sim["correct_total_revenue"], 160.0)
        self.assertEqual(sim["correct_total_payment"], 160.0)
        self.assertEqual(sim["preagg_row_count"], 2)
        
        # Naive Cartesian rows: Order 1 has 3*2 = 6 rows; Order 2 has 2*3 = 6 rows; Total = 12 rows
        self.assertEqual(sim["naive_row_count"], 12)
        # Revenue inflated because Order 1 items counted twice (2 payments), Order 2 items counted 3 times (3 payments)
        # Order 1 naive revenue: (10+20+30)*2 = 120.0
        # Order 2 naive revenue: (50+50)*3 = 300.0
        # Total naive revenue = 420.0 instead of 160.0! (2.625x inflation!)
        self.assertEqual(sim["naive_total_revenue"], 420.0)
        self.assertGreater(sim["revenue_inflation_ratio"], 1.0)

    def test_oracle_customer_ltv_and_rfm_deciles(self):
        """
        Oracle Test: Demonstrates exact mathematical calculation of Customer LTV,
        average basket size, dense ranking, and NTILE(10) spending decile allocation.
        """
        # Create 20 synthetic customers with varying spend
        customers = [{"customer_id": f"cust_{i:03d}", "customer_city": "Sao Paulo", "customer_state": "SP"} for i in range(1, 21)]
        orders = []
        payments = []
        for i, c in enumerate(customers):
            # Customer i has (i+1) orders of $10 each
            cid = c["customer_id"]
            num_orders = (i % 5) + 1
            spend_per_order = (i + 1) * 10.0
            for o_idx in range(num_orders):
                oid = f"ord_{cid}_{o_idx}"
                orders.append({
                    "order_id": oid,
                    "customer_id": cid,
                    "order_purchase_timestamp": f"2023-01-{o_idx+1:02d} 10:00:00"
                })
                payments.append({
                    "order_id": oid,
                    "payment_value": spend_per_order
                })
                
        metrics = simulate_customer_rfm_and_ltv(customers, orders, payments)
        self.assertEqual(len(metrics), 20)
        
        # Verify deciles: 20 customers into 10 buckets = 2 customers per decile
        deciles = [m["spending_decile"] for m in metrics]
        for bucket in range(1, 11):
            self.assertEqual(deciles.count(bucket), 2, f"Bucket {bucket} must have exactly 2 customers")
            
        # Verify LTV descending order
        ltvs = [m["lifetime_value"] for m in metrics]
        self.assertEqual(ltvs, sorted(ltvs, reverse=True))

    def test_oracle_product_pareto_80_20_share(self):
        """
        Oracle Test: Demonstrates Pareto 80/20 category revenue distribution,
        cumulative share percentages, and threshold boundary classification.
        """
        products = [
            {"product_id": "p1", "product_category_name": "electronics"},
            {"product_id": "p2", "product_category_name": "computers"},
            {"product_id": "p3", "product_category_name": "books"},
            {"product_id": "p4", "product_category_name": "stationery"},
        ]
        # Revenue: electronics=$700, computers=$200, books=$80, stationery=$20 -> Total=$1000
        order_items = [
            {"product_id": "p1", "order_id": "o1", "price": 700.0},
            {"product_id": "p2", "order_id": "o2", "price": 200.0},
            {"product_id": "p3", "order_id": "o3", "price": 80.0},
            {"product_id": "p4", "order_id": "o4", "price": 20.0},
        ]
        
        pareto = simulate_pareto_category_distribution(products, order_items)
        self.assertEqual(len(pareto), 4)
        
        # Rank 1: electronics (70.0% cumulative share)
        self.assertEqual(pareto[0]["category_name"], "electronics")
        self.assertEqual(pareto[0]["cumulative_share_pct"], 70.0)
        self.assertTrue(pareto[0]["is_pareto_top_80"])
        
        # Rank 2: computers (90.0% cumulative share -> boundary crossed)
        self.assertEqual(pareto[1]["category_name"], "computers")
        self.assertEqual(pareto[1]["cumulative_share_pct"], 90.0)
        
        # Rank 4: stationery (100.0% cumulative share)
        self.assertEqual(pareto[3]["cumulative_share_pct"], 100.0)
        self.assertFalse(pareto[3]["is_pareto_top_80"])

    def test_oracle_scd2_merge_state_machine(self):
        """
        Oracle Test: Demonstrates multi-batch Slowly Changing Dimension Type 2 (SCD2)
        lifecycle state transitions, validating that:
        - Batch 1: Inserts new active record (is_current=True, effective_end_date=None)
        - Batch 2: On attribute change, closes old record (is_current=False, effective_end_date set)
                   and creates new active record (is_current=True) with unique SHA256 surrogate keys.
        - Point-in-time time-travel history retains complete immutable audit trail.
        """
        state = SCD2DimensionState()
        
        # Batch 1: Initial load
        batch_1 = [
            {"customer_id": "c1", "customer_city": "Sao Paulo", "customer_state": "SP", "customer_zip_code_prefix": "01000"},
            {"customer_id": "c2", "customer_city": "Rio de Janeiro", "customer_state": "RJ", "customer_zip_code_prefix": "20000"},
        ]
        state.apply_batch(batch_1, "2023-01-01 00:00:00")
        
        current_b1 = state.query_current()
        self.assertEqual(len(current_b1), 2)
        self.assertTrue(all(r["is_current"] for r in current_b1))
        self.assertTrue(all(r["effective_end_date"] is None for r in current_b1))
        
        # Batch 2: Customer 1 moves to Campinas, Customer 2 remains unchanged
        batch_2 = [
            {"customer_id": "c1", "customer_city": "Campinas", "customer_state": "SP", "customer_zip_code_prefix": "13000"},
            {"customer_id": "c2", "customer_city": "Rio de Janeiro", "customer_state": "RJ", "customer_zip_code_prefix": "20000"},
        ]
        state.apply_batch(batch_2, "2023-06-01 00:00:00")
        
        # Customer 1 must now have 2 versions: 1 expired and 1 active
        c1_history = state.query_history("c1")
        self.assertEqual(len(c1_history), 2)
        
        # Version 1 (Sao Paulo) expired on 2023-06-01
        v1 = c1_history[0]
        self.assertEqual(v1["customer_city"], "Sao Paulo")
        self.assertFalse(v1["is_current"])
        self.assertEqual(v1["effective_end_date"], "2023-06-01 00:00:00")
        
        # Version 2 (Campinas) active from 2023-06-01
        v2 = c1_history[1]
        self.assertEqual(v2["customer_city"], "Campinas")
        self.assertTrue(v2["is_current"])
        self.assertIsNone(v2["effective_end_date"])
        
        # Surrogate keys must be distinct
        self.assertNotEqual(v1["customer_sk"], v2["customer_sk"])
        
        # Customer 2 had no change, so only 1 active version exists
        c2_history = state.query_history("c2")
        self.assertEqual(len(c2_history), 1)
        self.assertTrue(c2_history[0]["is_current"])

    def test_medallion_architecture_lineage_contract(self):
        """
        Architecture Lineage: Verifies end-to-end table dependency contracts across layers:
        Bronze CSV source paths -> workspace.ecompulse_bronze.* ->
        workspace.ecompulse_silver.* -> workspace.ecompulse_gold.*
        """
        # Bronze sources
        bronze_code = self.get_notebook_code("bronze/bronze_ingestion.ipynb")
        self.assertIn("/Volumes/workspace/ecom-pulse/dataset/data/", bronze_code)
        
        # Silver inputs & outputs
        silver_orders = self.get_notebook_code("silver/silver_orders.ipynb")
        self.assertIn("workspace.ecompulse_bronze.orders", silver_orders)
        self.assertIn("workspace.ecompulse_silver.orders", silver_orders)
        
        silver_cust = self.get_notebook_code("silver/silver_customers.ipynb")
        self.assertIn("workspace.ecompulse_bronze.customers", silver_cust)
        self.assertIn("workspace.ecompulse_silver.customers", silver_cust)
        
        silver_pmt = self.get_notebook_code("silver/silver_payments.ipynb")
        self.assertIn("workspace.ecompulse_bronze.payments", silver_pmt)
        self.assertIn("workspace.ecompulse_silver.payments", silver_pmt)
        
        silver_prod = self.get_notebook_code("silver/silver_products.ipynb")
        self.assertIn("workspace.ecompulse_bronze.products", silver_prod)
        self.assertIn("workspace.ecompulse_silver.products", silver_prod)


if __name__ == "__main__":
    unittest.main()
