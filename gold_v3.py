import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

SILVER_V3_DIR = Path("silver_data_v3")


def run_gold_load_v3() -> None:
    """
    V3 Gold Layer - Attention 500 Analytics Database Load
    
    Loads to NEW V3 tables:
    - channels_log_v3
    - videos_log_v3
    - trending_videos_log_v3
    
    This preserves your V1 and V2 tables and allows you to run
    all three pipeline versions simultaneously.
    
    NEW V3 COLUMNS IN VIDEO TABLES:
    - next_milestone (bigint)
    - views_to_next_milestone (bigint)
    - days_to_milestone (float, nullable)
    - is_approaching_milestone (boolean)
    - milestone_tier (text)
    - milestone_progress_pct (float)
    """
    load_dotenv()

    # Neon connection string from environment
    db_uri = os.getenv("NEON_DATABASE_URL")
    if not db_uri:
        raise RuntimeError("NEON_DATABASE_URL not set. Add it to .env or GitHub Secrets.")
    
    # Clean the connection string - remove 'psql' prefix if present
    db_uri = db_uri.strip()
    if db_uri.startswith("psql "):
        db_uri = db_uri[5:].strip()
    # Remove quotes if present
    db_uri = db_uri.strip("'\"")
    
    print(f"[GOLD-V3] Connecting to Neon database...")

    channels_path = SILVER_V3_DIR / "channels_v3.parquet"
    videos_path = SILVER_V3_DIR / "videos_v3.parquet"
    trending_path = SILVER_V3_DIR / "trending_videos_v3.parquet"

    # Load channels
    if channels_path.exists():
        df_channels = pl.read_parquet(channels_path)
        df_channels.write_database(
            table_name="channels_log_v3",
            connection=db_uri,
            if_table_exists="append",
        )
        print(f"[GOLD-V3] ✓ Loaded {len(df_channels)} channel records into channels_log_v3")
    else:
        print("[GOLD-V3] No channels_v3.parquet found, skipping channels.")

    # Load videos
    if videos_path.exists():
        df_videos = pl.read_parquet(videos_path)
        df_videos.write_database(
            table_name="videos_log_v3",
            connection=db_uri,
            if_table_exists="append",
        )
        print(f"[GOLD-V3] ✓ Loaded {len(df_videos)} video records into videos_log_v3")
    else:
        print("[GOLD-V3] No videos_v3.parquet found, skipping videos.")

    # Load trending videos
    if trending_path.exists():
        df_trending = pl.read_parquet(trending_path)
        df_trending.write_database(
            table_name="trending_videos_log_v3",
            connection=db_uri,
            if_table_exists="append",
        )
        print(f"[GOLD-V3] ✓ Loaded {len(df_trending)} trending video records into trending_videos_log_v3")
    else:
        print("[GOLD-V3] No trending_videos_v3.parquet found, skipping trending videos.")

    print("\n[GOLD-V3] ✅ All V3 data loaded to Neon database!")


if __name__ == "__main__":
    run_gold_load_v3()
