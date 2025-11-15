import json
from pathlib import Path
from datetime import datetime

import polars as pl

BRONZE_DIR = Path("bronze_data")
SILVER_DIR = Path("silver_data")


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def run_silver_transformation() -> None:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    channel_rows: list[dict] = []
    video_rows: list[dict] = []

    json_files = list(BRONZE_DIR.glob("**/*.json"))
    if not json_files:
        print("[SILVER] No JSON files found in bronze_data.")
        return

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[SILVER] Failed to read {path}: {e}")
            continue

        items = raw.get("items", [])
        ingest_ts = raw.get(
            "ingestion_timestamp",
            raw.get("ingest_timestamp", datetime.utcnow().isoformat()),
        )

        for item in items:
            kind = item.get("kind", "")

            # --- Channels ---
            if "channel" in kind:
                stats = item.get("statistics", {})
                snippet = item.get("snippet", {})
                channel_rows.append(
                    {
                        "channel_id": item.get("id"),
                        "channel_title": snippet.get("title"),
                        "description": snippet.get("description"),
                        "country": snippet.get("country"),
                        "published_at": snippet.get("publishedAt"),
                        "subscriber_count": _safe_int(
                            stats.get("subscriberCount")
                        ),
                        "view_count": _safe_int(stats.get("viewCount")),
                        "video_count": _safe_int(stats.get("videoCount")),
                        "ingest_timestamp": ingest_ts,
                    }
                )

            # --- Videos ---
            elif "video" in kind:
                stats = item.get("statistics", {})
                snippet = item.get("snippet", {})
                details = item.get("contentDetails", item.get("content_details", {}))

                video_rows.append(
                    {
                        "video_id": item.get("id"),
                        "channel_id": snippet.get("channelId"),
                        "video_title": snippet.get("title"),
                        "description": snippet.get("description"),
                        "published_at": snippet.get("publishedAt"),
                        "view_count": _safe_int(stats.get("viewCount")),
                        "like_count": _safe_int(stats.get("likeCount")),
                        "comment_count": _safe_int(stats.get("commentCount")),
                        "duration": details.get("duration"),
                        "ingest_timestamp": ingest_ts,
                    }
                )

    # --- Save to Parquet ---
    if channel_rows:
        pl.DataFrame(channel_rows).write_parquet(SILVER_DIR / "channels.parquet")
        print(
            f"[SILVER] Saved {len(channel_rows)} channel rows to silver_data/channels.parquet"
        )
    else:
        print("[SILVER] No channel rows to save.")

    if video_rows:
        pl.DataFrame(video_rows).write_parquet(SILVER_DIR / "videos.parquet")
        print(
            f"[SILVER] Saved {len(video_rows)} video rows to silver_data/videos.parquet"
        )
    else:
        print("[SILVER] No video rows to save.")


if __name__ == "__main__":
    run_silver_transformation()
