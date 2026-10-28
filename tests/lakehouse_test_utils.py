"""
Lakehouse Test Utilities & Algorithmic Oracles.

Provides robust utilities for inspecting Databricks Jupyter notebooks,
parsing ASTs, verifying Unity Catalog namespaces, checking window functions,
and executing mathematical reference simulations (Oracles) for Lakehouse pipelines.
"""

import ast
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple


def get_repo_root() -> str:
    """Return the absolute path to the repository root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def load_notebook(relative_path: str) -> Dict[str, Any]:
    """
    Load and parse a Jupyter notebook from relative path to repository root.
    
    Raises:
        FileNotFoundError: If the notebook does not exist.
        ValueError: If the file is not valid JSON.
    """
    repo_root = get_repo_root()
    abs_path = os.path.join(repo_root, relative_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Notebook not found at: {abs_path}")
    
    with open(abs_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in notebook {relative_path}: {e}")


def extract_code_cells(nb: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all code cells from a notebook dictionary."""
    cells = nb.get("cells", [])
    return [cell for cell in cells if cell.get("cell_type") == "code"]


def get_cell_text(cell: Dict[str, Any]) -> str:
    """Extract raw source string from a cell."""
    source = cell.get("source", [])
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def extract_python_and_sql(nb: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Separate Python code lines and Databricks `%sql` cells from a notebook.
    
    Returns:
        (python_code_string, list_of_sql_cell_strings)
    """
    py_chunks: List[str] = []
    sql_chunks: List[str] = []
    
    for cell in extract_code_cells(nb):
        text = get_cell_text(cell)
        stripped = text.strip()
        if not stripped:
            continue
            
        if stripped.startswith("%sql"):
            # Databricks SQL magic cell
            sql_chunks.append(text)
        else:
            # Python cell (may contain line-magics like %pip or %run)
            sanitized_lines = []
            for line in text.splitlines():
                if line.strip().startswith("%"):
                    sanitized_lines.append(f"# [MAGIC] {line}")
                else:
                    sanitized_lines.append(line)
            py_chunks.append("\n".join(sanitized_lines))
            
    return "\n\n".join(py_chunks), sql_chunks


def parse_python_ast(python_code: str, context_name: str = "<notebook>") -> ast.AST:
    """
    Parse Python code string into an AST module.
    
    Raises:
        SyntaxError: If python_code contains invalid Python syntax.
    """
    try:
        return ast.parse(python_code, filename=context_name)
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in {context_name} at line {e.lineno}: {e.msg}\n{e.text}")


def extract_table_references(code_text: str) -> Set[str]:
    """
    Extract table references from PySpark code text.
    Looks for patterns like:
    - .table("workspace.ecompulse_...")
    - .saveAsTable("workspace.ecompulse_...")
    - "workspace.ecompulse_..."
    """
    pattern = r'["\'](workspace\.ecompulse_[a-z0-9_]+\.[a-z0-9_]+)["\']'
    matches = re.findall(pattern, code_text, flags=re.IGNORECASE)
    return set(matches)


def check_fan_out_mitigation_ast(code_text: str) -> Dict[str, Any]:
    """
    Verify whether order_items and payments are pre-aggregated by order_id
    before joining with orders.
    
    Returns a dict with:
        - pre_aggregates_items: bool
        - pre_aggregates_payments: bool
        - joins_with_orders: bool
        - evidence: List[str]
    """
    evidence = []
    
    # Check for order_items aggregation by order_id
    items_agg_patterns = [
        r'order_items.*(?:groupBy|groupby)\s*\(\s*["\']order_id["\']',
        r'items_agg.*(?:groupBy|groupby)\s*\(\s*["\']order_id["\']',
        r'(?:groupBy|groupby)\s*\(\s*["\']order_id["\']\s*\).*(?:sum|count|agg)',
        r'agg_items\s*='
    ]
    items_pre_agg = any(re.search(pat, code_text, re.IGNORECASE | re.DOTALL) for pat in items_agg_patterns)
    if items_pre_agg:
        evidence.append("Found order_items pre-aggregation on order_id grain")
        
    # Check for payments aggregation by order_id
    payments_agg_patterns = [
        r'payments.*(?:groupBy|groupby)\s*\(\s*["\']order_id["\']',
        r'payments_agg.*(?:groupBy|groupby)\s*\(\s*["\']order_id["\']',
        r'agg_payments\s*='
    ]
    payments_pre_agg = any(re.search(pat, code_text, re.IGNORECASE | re.DOTALL) for pat in payments_agg_patterns)
    if payments_pre_agg:
        evidence.append("Found payments pre-aggregation on order_id grain")
        
    joins_orders = bool(re.search(r'\.join\s*\(.*orders.*["\']order_id["\']', code_text, re.IGNORECASE) or
                       re.search(r'orders.*\.join\s*\(.*["\']order_id["\']', code_text, re.IGNORECASE))
    if joins_orders:
        evidence.append("Found order join on order_id")
        
    return {
        "pre_aggregates_items": items_pre_agg,
        "pre_aggregates_payments": payments_pre_agg,
        "joins_with_orders": joins_orders,
        "evidence": evidence
    }


# ==============================================================================
# Algorithmic & Mathematical Oracles (Reference Models)
# ==============================================================================

def simulate_fan_out_cartesian_vs_preagg(
    orders: List[Dict[str, Any]],
    order_items: List[Dict[str, Any]],
    payments: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Mathematical Oracle simulating the Fan-Out problem in E-Commerce Data Marts.
    
    In the Olist schema:
    - 1 order can have N items (order_id, order_item_id)
    - 1 order can have M payment installments/vouchers (order_id, payment_sequential)
    
    If joined naively: orders JOIN order_items JOIN payments on order_id:
    - Total rows per order = N * M
    - Directly summing item price yields: sum(price) * M (inflated by M times!)
    - Directly summing payment_value yields: sum(payment_value) * N (inflated by N times!)
    
    With Pre-Aggregation:
    - items pre-aggregated to (order_id, total_items_revenue)
    - payments pre-aggregated to (order_id, total_payment_value)
    - orders JOIN items_agg JOIN payments_agg yields: 1 row per order, exact sums!
    """
    # 1. Naive Cartesian Join Simulation
    naive_rows = []
    for order in orders:
        oid = order["order_id"]
        o_items = [it for it in order_items if it["order_id"] == oid]
        o_payments = [pm for pm in payments if pm["order_id"] == oid]
        for it in o_items:
            for pm in o_payments:
                naive_rows.append({
                    "order_id": oid,
                    "item_id": it["order_item_id"],
                    "item_price": it["price"],
                    "payment_sequential": pm["payment_sequential"],
                    "payment_value": pm["payment_value"],
                })
                
    naive_total_revenue = sum(r["item_price"] for r in naive_rows)
    naive_total_payment = sum(r["payment_value"] for r in naive_rows)
    
    # 2. Pre-Aggregated Join Simulation (Correct Lakehouse Pattern)
    items_by_order: Dict[str, float] = {}
    items_count_by_order: Dict[str, int] = {}
    for it in order_items:
        oid = it["order_id"]
        items_by_order[oid] = items_by_order.get(oid, 0.0) + it["price"]
        items_count_by_order[oid] = items_count_by_order.get(oid, 0) + 1
        
    payments_by_order: Dict[str, float] = {}
    for pm in payments:
        oid = pm["order_id"]
        payments_by_order[oid] = payments_by_order.get(oid, 0.0) + pm["payment_value"]
        
    preagg_rows = []
    for order in orders:
        oid = order["order_id"]
        rev = items_by_order.get(oid, 0.0)
        cnt = items_count_by_order.get(oid, 0)
        pmt = payments_by_order.get(oid, 0.0)
        preagg_rows.append({
            "order_id": oid,
            "order_status": order.get("order_status"),
            "order_revenue": round(rev, 2),
            "items_count": cnt,
            "total_payment": round(pmt, 2)
        })
        
    correct_total_revenue = round(sum(r["order_revenue"] for r in preagg_rows), 2)
    correct_total_payment = round(sum(r["total_payment"] for r in preagg_rows), 2)
    
    return {
        "naive_row_count": len(naive_rows),
        "preagg_row_count": len(preagg_rows),
        "naive_total_revenue": round(naive_total_revenue, 2),
        "correct_total_revenue": correct_total_revenue,
        "naive_total_payment": round(naive_total_payment, 2),
        "correct_total_payment": correct_total_payment,
        "revenue_inflation_ratio": round(naive_total_revenue / correct_total_revenue, 4) if correct_total_revenue > 0 else 1.0,
        "payment_inflation_ratio": round(naive_total_payment / correct_total_payment, 4) if correct_total_payment > 0 else 1.0
    }


def simulate_customer_rfm_and_ltv(
    customers: List[Dict[str, Any]],
    orders: List[Dict[str, Any]],
    payments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Mathematical Oracle simulating Customer LTV, order sequencing,
    dense rank spending, and NTILE(10) decile allocation.
    """
    # Group payments by order_id
    pmt_by_order: Dict[str, float] = {}
    for p in payments:
        oid = p["order_id"]
        pmt_by_order[oid] = pmt_by_order.get(oid, 0.0) + p["payment_value"]
        
    # Associate orders with customer
    orders_by_cust: Dict[str, List[Dict[str, Any]]] = {}
    for o in orders:
        cid = o["customer_id"]
        o_copy = dict(o)
        o_copy["order_total"] = pmt_by_order.get(o["order_id"], 0.0)
        orders_by_cust.setdefault(cid, []).append(o_copy)
        
    results = []
    for c in customers:
        cid = c["customer_id"]
        c_orders = orders_by_cust.get(cid, [])
        # Sort orders by purchase timestamp for ROW_NUMBER() order sequence
        sorted_orders = sorted(c_orders, key=lambda x: x.get("order_purchase_timestamp", ""))
        
        total_spend = sum(o["order_total"] for o in sorted_orders)
        total_orders = len(sorted_orders)
        aov = round(total_spend / total_orders, 2) if total_orders > 0 else 0.0
        
        first_date = sorted_orders[0]["order_purchase_timestamp"] if sorted_orders else None
        last_date = sorted_orders[-1]["order_purchase_timestamp"] if sorted_orders else None
        
        results.append({
            "customer_id": cid,
            "customer_city": c.get("customer_city"),
            "customer_state": c.get("customer_state"),
            "total_orders": total_orders,
            "lifetime_value": round(total_spend, 2),
            "average_basket_size": aov,
            "first_purchase_date": first_date,
            "last_purchase_date": last_date,
        })
        
    # Compute DENSE_RANK() and NTILE(10) over lifetime_value descending
    results.sort(key=lambda x: x["lifetime_value"], reverse=True)
    
    current_rank = 1
    for i, row in enumerate(results):
        if i > 0 and row["lifetime_value"] < results[i - 1]["lifetime_value"]:
            current_rank += 1
        row["spend_dense_rank"] = current_rank
        
    # NTILE(10) distribution: partition N into 10 buckets
    n = len(results)
    if n > 0:
        base_size = n // 10
        remainder = n % 10
        start = 0
        for bucket in range(1, 11):
            size = base_size + (1 if bucket <= remainder else 0)
            for j in range(start, start + size):
                if j < n:
                    results[j]["spending_decile"] = bucket
            start += size
            
    return results


def simulate_pareto_category_distribution(
    products: List[Dict[str, Any]],
    order_items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Mathematical Oracle simulating Product Category Pareto Analysis (80/20 rule).
    Calculates gross revenue per category, dense rank, running total, and percentage share.
    """
    # Map product to category
    prod_cat = {p["product_id"]: p.get("product_category_name", "unknown") for p in products}
    
    category_metrics: Dict[str, Dict[str, Any]] = {}
    for item in order_items:
        pid = item["product_id"]
        cat = prod_cat.get(pid, "unknown")
        if cat not in category_metrics:
            category_metrics[cat] = {
                "category_name": cat,
                "total_units_sold": 0,
                "gross_revenue": 0.0,
                "order_ids": set()
            }
        category_metrics[cat]["total_units_sold"] += 1
        category_metrics[cat]["gross_revenue"] += item.get("price", 0.0)
        category_metrics[cat]["order_ids"].add(item.get("order_id"))
        
    rows = []
    for cat, data in category_metrics.items():
        rows.append({
            "category_name": cat,
            "total_units_sold": data["total_units_sold"],
            "gross_revenue": round(data["gross_revenue"], 2),
            "order_count": len(data["order_ids"])
        })
        
    rows.sort(key=lambda x: x["gross_revenue"], reverse=True)
    total_revenue = sum(r["gross_revenue"] for r in rows)
    
    running_revenue = 0.0
    current_rank = 1
    for i, r in enumerate(rows):
        if i > 0 and r["gross_revenue"] < rows[i - 1]["gross_revenue"]:
            current_rank += 1
        r["category_revenue_rank"] = current_rank
        running_revenue += r["gross_revenue"]
        r["running_total_revenue"] = round(running_revenue, 2)
        share = (running_revenue / total_revenue * 100.0) if total_revenue > 0 else 0.0
        r["cumulative_share_pct"] = round(share, 2)
        r["is_pareto_top_80"] = (share <= 80.0) or (i == 0) # Top 80% contributors
        
    return rows


def generate_scd2_surrogate_key(natural_key: str, start_date: str) -> str:
    """Generate SHA256 surrogate key from natural key and effective start date."""
    content = f"{natural_key}_{start_date}".encode("utf-8")
    return hashlib.sha256(content).hexdigest()


class SCD2DimensionState:
    """
    Stateful Oracle simulating Delta Lake Slowly Changing Dimension Type 2 (SCD2).
    Validates staged atomic MERGE operations across multiple batches.
    """
    def __init__(self):
        # Store versions: list of dicts
        self.rows: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.version = 0

    def apply_batch(self, incoming_records: List[Dict[str, Any]], batch_timestamp: str):
        """
        Simulate Delta Lake staged MERGE for SCD Type 2:
        1. Find active rows in target matching source on customer_id
        2. Detect changes in tracked attributes (city, state, zip_code)
        3. Close out changed records (is_current=False, effective_end_date=batch_timestamp)
        4. Insert new version (is_current=True, effective_start_date=batch_timestamp, effective_end_date=None)
        5. Insert new customers directly
        """
        self.version += 1
        action_summary = {"version": self.version, "timestamp": batch_timestamp, "inserted": 0, "updated": 0}
        
        active_lookup = {r["customer_id"]: r for r in self.rows if r["is_current"]}
        
        for inc in incoming_records:
            cid = inc["customer_id"]
            if cid in active_lookup:
                curr = active_lookup[cid]
                # Check if tracked attributes changed
                changed = (
                    curr["customer_city"] != inc.get("customer_city") or
                    curr["customer_state"] != inc.get("customer_state") or
                    curr["customer_zip_code_prefix"] != inc.get("customer_zip_code_prefix")
                )
                if changed:
                    # Close out current record
                    curr["is_current"] = False
                    curr["effective_end_date"] = batch_timestamp
                    action_summary["updated"] += 1
                    
                    # Create new version
                    new_sk = generate_scd2_surrogate_key(cid, batch_timestamp)
                    new_rec = {
                        "customer_sk": new_sk,
                        "customer_id": cid,
                        "customer_city": inc.get("customer_city"),
                        "customer_state": inc.get("customer_state"),
                        "customer_zip_code_prefix": inc.get("customer_zip_code_prefix"),
                        "effective_start_date": batch_timestamp,
                        "effective_end_date": None,
                        "is_current": True
                    }
                    self.rows.append(new_rec)
                    action_summary["inserted"] += 1
            else:
                # Brand new record
                sk = generate_scd2_surrogate_key(cid, batch_timestamp)
                new_rec = {
                    "customer_sk": sk,
                    "customer_id": cid,
                    "customer_city": inc.get("customer_city"),
                    "customer_state": inc.get("customer_state"),
                    "customer_zip_code_prefix": inc.get("customer_zip_code_prefix"),
                    "effective_start_date": batch_timestamp,
                    "effective_end_date": None,
                    "is_current": True
                }
                self.rows.append(new_rec)
                action_summary["inserted"] += 1
                
        self.history.append(action_summary)

    def query_current(self) -> List[Dict[str, Any]]:
        """Return currently active records (is_current == True)."""
        return [r for r in self.rows if r["is_current"]]

    def query_history(self, customer_id: str) -> List[Dict[str, Any]]:
        """Return complete historical timeline for a given customer_id."""
        return [r for r in self.rows if r["customer_id"] == customer_id]
