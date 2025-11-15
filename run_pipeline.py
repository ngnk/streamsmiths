"""
One-button runner for the YouTube pipeline:
BRONZE -> SILVER -> GOLD
"""

from dotenv import load_dotenv

from bronze import run_bronze_ingestion
from silver import run_silver_transformation
from gold import run_gold_load


def main():
    # Load environment variables from .env
    load_dotenv()

    print("\n=== STEP 1: BRONZE (YouTube -> JSON) ===")
    run_bronze_ingestion()

    print("\n=== STEP 2: SILVER (JSON -> Parquet) ===")
    run_silver_transformation()

    print("\n=== STEP 3: GOLD (Parquet -> Neon) ===")
    run_gold_load()

    print("\n✅ Pipeline completed successfully (BRONZE → SILVER → GOLD).")


if __name__ == "__main__":
    main()
