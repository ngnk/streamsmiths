import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

SILVER_V2_DIR = Path("silver_data_v2")


def run_gold_load_v2() -> None:
    """
    Enhanced gold layer V2 - loads to separate tables:
    - channels_log_v2
    - videos_log_v2
    - trending_videos_log_v2
    
    This preserves your existing tables and allows you to compare
    the old vs new pipeline approaches.
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
    
    print(f"[GOLD-V2] Connecting to Neon database...")

    channels_path = SILVER_V2_DIR / "channels_v2.parquet"
    videos_path = SILVER_V2_DIR / "videos_v2.parquet"
    trending_path = SILVER_V2_DIR / "trending_videos_v2.parquet"

    # Load channels
    if channels_path.exists():
        df_channels = pl.read_parquet(channels_path)
        df_channels.write_database(
            table_name="channels_log_v2",
            connection=db_uri,
            if_table_exists="append",
        )
        print(f"[GOLD-V2] ✓ Loaded {len(df_channels)} channel records into channels_log_v2")
    else:
        print("[GOLD-V2] No channels_v2.parquet found, skipping channels.")

    # Load videos
    if videos_path.exists():
        df_videos = pl.read_parquet(videos_path)
        df_videos.write_database(
            table_name="videos_log_v2",
            connection=db_uri,
            if_table_exists="append",
        )
        print(f"[GOLD-V2] ✓ Loaded {len(df_videos)} video records into videos_log_v2")
    else:
        print("[GOLD-V2] No videos_v2.parquet found, skipping videos.")

    # Load trending videos
    if trending_path.exists():
        df_trending = pl.read_parquet(trending_path)
        df_trending.write_database(
            table_name="trending_videos_log_v2",
            connection=db_uri,
            if_table_exists="append",
        )
        print(f"[GOLD-V2] ✓ Loaded {len(df_trending)} trending video records into trending_videos_log_v2")
    else:
        print("[GOLD-V2] No trending_videos_v2.parquet found, skipping trending videos.")

    print("\n[GOLD-V2] ✅ All data loaded to Neon database!")


if __name__ == "__main__":
    run_gold_load_v2()
