import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

SILVER_DIR = Path("silver_data")


def run_gold_load() -> None:
    load_dotenv()

    # Neon connection string from environment
    db_uri = os.getenv("NEON_DATABASE_URL")
    if not db_uri:
        raise RuntimeError("NEON_DATABASE_URL not set. Add it to .env file.")
    
    # Clean the connection string - remove 'psql' prefix if present
    db_uri = db_uri.strip()
    if db_uri.startswith("psql "):
        db_uri = db_uri[5:].strip()
    # Remove quotes if present
    db_uri = db_uri.strip("'\"")
    
    print(f"[GOLD] Connecting to Neon database...")

    channels_path = SILVER_DIR / "channels.parquet"
    videos_path = SILVER_DIR / "videos.parquet"

    if channels_path.exists():
        df_channels = pl.read_parquet(channels_path)
        df_channels.write_database(
            table_name="channels_log",
            connection=db_uri,
            if_table_exists="append",
        )
        print(
            f"[GOLD] Loaded {len(df_channels)} channel records into channels_log."
        )
    else:
        print("[GOLD] No channels.parquet found, skipping channels.")

    if videos_path.exists():
        df_videos = pl.read_parquet(videos_path)
        df_videos.write_database(
            table_name="videos_log",
            connection=db_uri,
            if_table_exists="append",
        )
        print(f"[GOLD] Loaded {len(df_videos)} video records into videos_log.")
    else:
        print("[GOLD] No videos.parquet found, skipping videos.")


if __name__ == "__main__":
    run_gold_load()
