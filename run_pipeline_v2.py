"""
One-button runner for the Enhanced YouTube Pipeline V2:
BRONZE-V2 -> SILVER-V2 -> GOLD-V2

This version includes:
- Comprehensive channel metadata (custom URLs, thumbnails, keywords, topics)
- Enhanced video metrics (engagement rate, views per day, duration analysis)
- Trending videos capture
- Clean timestamp formatting (YYYY-MM-DD HH:MM:SS)
- Separate tables to preserve existing progress
"""

from dotenv import load_dotenv

from bronze_v2 import run_bronze_ingestion_v2
from silver_v2 import run_silver_transformation_v2
from gold_v2 import run_gold_load_v2


def main():
    # Load environment variables from .env
    load_dotenv()

    print("\n" + "="*60)
    print("ENHANCED YOUTUBE PIPELINE V2")
    print("="*60)

    print("\n=== STEP 1: BRONZE-V2 (YouTube API -> JSON) ===")
    print("Fetching channel details + top videos + trending videos...")
    run_bronze_ingestion_v2()

    print("\n=== STEP 2: SILVER-V2 (JSON -> Parquet) ===")
    print("Transforming data with enhanced metrics...")
    run_silver_transformation_v2()

    print("\n=== STEP 3: GOLD-V2 (Parquet -> Neon) ===")
    print("Loading to database tables (channels_log_v2, videos_log_v2, trending_videos_log_v2)...")
    run_gold_load_v2()

    print("\n" + "="*60)
    print("✅ PIPELINE V2 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNew tables created in your Neon database:")
    print("  • channels_log_v2")
    print("  • videos_log_v2")
    print("  • trending_videos_log_v2")
    print("\nYour original tables (channels_log, videos_log) remain untouched.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
