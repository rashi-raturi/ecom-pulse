## 01

Batch ETL pipeline — start here

Current priority

- Read raw order and customer CSV files.
- Clean missing values, duplicates, and invalid records.
- Join orders, customers, and products.
- Use aggregations and window functions for analytics.
- Write cleaned datasets to Parquet.
- Organize transformations into reusable Python functions.

  Stack: PySpark + Databricks + Parquet

## 02

Build a Lakehouse

- Introduce Delta Lake tables.
- Implement Bronze, Silver, and Gold layers.
- Handle incremental data loads.
- Implement Slowly Changing Dimensions (SCD Type 2).
- Learn schema enforcement, time travel, and `MERGE`.

  Stack: PySpark + Delta Lake

## 03

Add streaming and orchestration

- Generate order events with Kafka.
- Process events using Spark Structured Streaming.
- Schedule and monitor jobs with Airflow.
- Handle late-arriving and duplicate events.

  Stack: Kafka + Spark Streaming + Airflow

## 04

Make it production-ready

- Add automated data-quality checks.
- Containerize the project with Docker.
- Create a dimensional warehouse with a star schema.
- Expose selected analytics through FastAPI.
- Add logging, tests, and documentation.

  Stack: PostgreSQL + Docker + FastAPI
