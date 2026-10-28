# Ecommerce Data Pipeline: Medallion Lakehouse

End-to-end Lakehouse batch ETL pipeline implementing the Medallion architecture (Bronze, Silver, Gold) and Slowly Changing Dimensions (SCD Type 2) using PySpark and Delta Lake on Databricks Unity Catalog.

## Architecture and Data Flow

```mermaid
flowchart LR
    subgraph Storage["Storage Layer: Databricks Volumes"]
        direction TB
        CSV_ORD["olist_orders_dataset.csv"]
        CSV_CUST["olist_customers_dataset.csv"]
        CSV_ITEM["olist_order_items_dataset.csv"]
        CSV_PAY["olist_order_payments_dataset.csv"]
        CSV_PROD["olist_products_dataset.csv"]
    end

    subgraph Bronze["Bronze Layer: workspace.ecompulse_bronze (Raw Delta)"]
        direction TB
        B_ORD["orders (raw + ingested_at)"]
        B_CUST["customers (raw + ingested_at)"]
        B_ITEM["order_items (raw + ingested_at)"]
        B_PAY["payments (raw + ingested_at)"]
        B_PROD["products (raw + ingested_at)"]
    end

    subgraph Silver["Silver Layer: workspace.ecompulse_silver (Cleaned Delta)"]
        direction TB
        S_ORD["orders (typed, validated timestamps, status)"]
        S_CUST["customers (deduplicated, whitespace-trimmed)"]
        S_ITEM["order_items (price, freight cast)"]
        S_PAY["payments (payment types, installment checks)"]
        S_PROD["products (standardized attributes)"]
    end

    subgraph Gold["Gold Layer: workspace.ecompulse_gold (Analytical Marts & Dimensions)"]
        direction TB
        G_SALES["daily_sales_analytics & monthly_sales_analytics (Running revenue, 7d moving avg, MoM growth)"]
        G_CUST["customer_analytics (CLV, RFM spend deciles, order sequences)"]
        G_PROD["product_performance & category_performance (Pareto 80/20, revenue ranking)"]
        G_SCD2["dim_customers_scd2 (SCD Type 2: SHA-256 keys, effective dates, is_current)"]
    end

    subgraph Consumers["Downstream Serving & Analytics"]
        direction TB
        BI_SALES["Sales Dashboards & KPI Reports"]
        BI_CRM["Marketing & Retention Cohorts"]
        BI_CAT["Merchandising & Inventory Analysis"]
        AUDIT["Audit & Historical Point-in-Time Queries"]
    end

    CSV_ORD --> B_ORD
    CSV_CUST --> B_CUST
    CSV_ITEM --> B_ITEM
    CSV_PAY --> B_PAY
    CSV_PROD --> B_PROD

    B_ORD -->|Cleaning & Deduplication| S_ORD
    B_CUST -->|Trimming & Standardization| S_CUST
    B_ITEM -->|Type Casting & Schema Check| S_ITEM
    B_PAY -->|Validation & Normalization| S_PAY
    B_PROD -->|Attribute Standardization| S_PROD

    S_ORD & S_ITEM & S_PAY -->|Order-Grain Pre-aggregation & Windowing| G_SALES
    S_CUST & S_ORD & S_PAY -->|Customer Aggregation & RFM Windowing| G_CUST
    S_PROD & S_ITEM & S_ORD -->|Product Aggregations & Pareto 80/20| G_PROD
    S_CUST -->|Staged Atomic Delta MERGE| G_SCD2

    G_SALES --> BI_SALES
    G_CUST --> BI_CRM
    G_PROD --> BI_CAT
    G_SCD2 --> AUDIT
```

## Directory Structure

```text
Ecommerce Data Pipeline/
├── bronze/
│   └── bronze_ingestion.ipynb        # Ingests raw CSVs into Bronze Delta tables
├── silver/
│   ├── silver_customers.ipynb       # Customer cleaning and deduplication
│   ├── silver_orders.ipynb          # Order timestamps and status normalization
│   ├── silver_order_items.ipynb     # Line item calculations and type casting
│   ├── silver_payments.ipynb        # Payment methods and installment validation
│   └── silver_products.ipynb        # Product dimension standardizations
├── gold/
│   ├── gold_sales_analytics.ipynb   # Daily/monthly revenue, 7d moving avg, MoM growth
│   ├── gold_customer_analytics.ipynb# Customer LTV, order tenure, RFM deciles
│   ├── gold_product_analytics.ipynb # Item volumes, category Pareto 80/20 analysis
│   └── gold_scd2_dimensions.ipynb   # dim_customers_scd2 Delta MERGE and time travel
└── tests/
    ├── run_e2e_tests.py             # 4-tier E2E validation runner
    ├── test_e2e_lakehouse.py        # Schema, window function, and MERGE tests
    └── lakehouse_test_utils.py      # Notebook AST parser and Spark mocks
```

## Data Models

```mermaid
erDiagram
    dim_customers_scd2 {
        string customer_sk PK
        string customer_id
        int customer_zip_code_prefix
        string customer_city
        string customer_state
        timestamp effective_start_date
        timestamp effective_end_date
        boolean is_current
    }

    customer_analytics {
        string customer_id PK
        int total_orders
        double total_spend
        double avg_order_value
        int customer_tenure_days
        int rfm_spend_decile
    }

    daily_sales_analytics {
        date order_date PK
        int total_orders
        double gross_revenue
        double running_total_revenue
        double rolling_7d_avg_revenue
        double dod_revenue_growth_rate
    }

    product_performance {
        string product_id PK
        string product_category_name
        int total_units_sold
        double gross_revenue
        double pareto_cumulative_revenue_pct
    }
```

## Pipeline Layers

### Bronze (`workspace.ecompulse_bronze`)
- Ingests source CSV files from volume storage into append/overwrite Delta tables.
- Preserves raw schema and tracks ingestion metadata (`ingested_at`).

### Silver (`workspace.ecompulse_silver`)
- Type casting, whitespace trimming, and null handling.
- Timestamp derivations and order status standardization.
- Deduplication on primary keys (`order_id`, `customer_id`, `product_id`).

### Gold (`workspace.ecompulse_gold`)
- **Sales Analytics**: Daily and monthly aggregations with anti-fanout pre-aggregation. Uses PySpark window functions for running revenue totals, 7-day rolling moving averages, and DoD / MoM growth rates.
- **Customer 360**: Computes Customer Lifetime Value (CLV), purchase frequency, tenure, and RFM spend deciles via `NTILE(10)` and `DENSE_RANK()`.
- **Product & Category Performance**: Aggregates unit sales, revenue volume, and category share via cumulative Pareto 80/20 distribution.

### Slowly Changing Dimensions (SCD Type 2)

```mermaid
flowchart TD
    SRC["Silver Customers Update Batch"] --> CHK{"Record exists & attributes changed?"}
    CHK -- "Yes (Change detected)" --> EXP["Expire current record: is_current = false, effective_end_date = batch_timestamp"]
    EXP --> INS["Insert new active version: customer_sk = sha256(customer_id + batch_timestamp), is_current = true, effective_start_date = batch_timestamp"]
    CHK -- "No (New record)" --> INS
```

- Target table: `workspace.ecompulse_gold.dim_customers_scd2`.
- Tracks location changes (`customer_city`, `customer_state`, `customer_zip_code_prefix`).
- Staged atomic Delta `MERGE` (`whenMatchedUpdate` for expired versions, `whenNotMatchedInsert` for new entries).
- Watermark incremental loading to process only updated records.
- Delta Lake features: Schema enforcement, schema evolution (`mergeSchema`), time travel (`VERSION AS OF` / `TIMESTAMP AS OF`), and table audit (`DESCRIBE HISTORY`).
