from dotenv import load_dotenv

from bronze_v3 import run_bronze_ingestion_v3
from silver_v3 import run_silver_transformation_v3
from gold_v3 import run_gold_load_v3


def main():
    # Load environment variables from .env
    load_dotenv()

    print("\n" + "="*70)
    print("🚀 ATTENTION 500 YOUTUBE PIPELINE V3 🚀")
    print("="*70)
    print("\n📊 V3 Features:")
    print("  • Multi-tier milestone tracking (10M → 1B+)")
    print("  • 5% approaching threshold detection")
    print("  • Days to milestone estimates")
    print("  • Joseph's ratio metrics & attention segments")
    print("  • Separate V3 tables (V1 & V2 preserved)")
    print("="*70)

    print("\n=== STEP 1: BRONZE-V3 (YouTube API → JSON) ===")
    print("Fetching channel details + top videos + trending videos...")
    run_bronze_ingestion_v3()

    print("\n=== STEP 2: SILVER-V3 (JSON → Parquet) ===")
    print("Transforming data with V3 milestone tracking...")
    run_silver_transformation_v3()

    print("\n=== STEP 3: GOLD-V3 (Parquet → Neon) ===")
    print("Loading to V3 database tables...")
    run_gold_load_v3()

    print("\n" + "="*70)
    print("✅ PIPELINE V3 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n📊 New V3 tables in your Neon database:")
    print("  • channels_log_v3")
    print("  • videos_log_v3 (with milestone tracking!)")
    print("  • trending_videos_log_v3 (with milestone tracking!)")
    print("\n💾 Your V1 and V2 tables remain untouched.")
    print("\n🎯 Ready for dashboard analytics and ML models!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
