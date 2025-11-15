# dags/youtube_pipeline.py
from datetime import datetime, timedelta

from airflow.models import DAG
from airflow.operators.python import PythonOperator

from project_scripts.bronze import run_bronze_ingestion, get_channel_ids
from project_scripts.silver import run_silver_transformation
from project_scripts.gold import run_gold_load

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="youtube_bronze_silver_gold",
    description="YouTube Bronze-Silver-Gold pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@hourly",
    catchup=False,
    max_active_runs=1,
) as dag:

    bronze_task = PythonOperator(
        task_id="bronze_ingest_channels",
        python_callable=run_bronze_ingestion,
        op_kwargs={"channel_ids": get_channel_ids()},
    )

    silver_task = PythonOperator(
        task_id="silver_transform_json_to_parquet",
        python_callable=run_silver_transformation,
    )

    gold_task = PythonOperator(
        task_id="gold_load_parquet_to_neon",
        python_callable=run_gold_load,
    )

    bronze_task >> silver_task >> gold_task
