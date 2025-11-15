import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import polars as pl

BRONZE_V2_DIR = Path("bronze_data_v2")
SILVER_V2_DIR = Path("silver_data_v2")


def _safe_int(val, default: int = 0) -> int:
    """Safely convert to int"""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert to float"""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _parse_iso_timestamp(iso_str: str) -> str:
    """
    Parse ISO timestamp and return clean UTC timestamp string.
    Converts: "2025-11-15T02:13:05.493325" -> "2025-11-15 02:13:05"
    """
    if not iso_str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Handle various ISO formats
        if "+" in iso_str:
            dt = datetime.fromisoformat(iso_str)
        elif "Z" in iso_str:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(iso_str)
        
        # Convert to UTC if not already
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        
        # Return clean format without microseconds
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _extract_thumbnail_url(thumbnails: dict, quality: str = "high") -> Optional[str]:
    """Extract thumbnail URL from thumbnails dict, preferring high quality"""
    if not thumbnails:
        return None
    
    # Try in order of preference
    for q in [quality, "high", "medium", "default"]:
        if q in thumbnails:
            return thumbnails[q].get("url")
    
    return None


def _parse_duration_to_seconds(duration: str) -> int:
    """
    Convert ISO 8601 duration (PT1H2M30S) to total seconds.
    Useful for analyzing video length.
    """
    if not duration or not duration.startswith("PT"):
        return 0
    
    duration = duration[2:]  # Remove 'PT'
    hours = minutes = seconds = 0
    
    try:
        if "H" in duration:
            hours, duration = duration.split("H")
            hours = int(hours)
        
        if "M" in duration:
            minutes, duration = duration.split("M")
            minutes = int(minutes)
        
        if "S" in duration:
            seconds = int(duration.replace("S", ""))
        
        return hours * 3600 + minutes * 60 + seconds
    except Exception:
        return 0


def _calculate_engagement_rate(likes: int, comments: int, views: int) -> float:
    """Calculate engagement rate as (likes + comments) / views"""
    if views == 0:
        return 0.0
    return ((likes + comments) / views) * 100


def _calculate_days_since_publish(published_at: str) -> int:
    """Calculate days since video was published"""
    if not published_at:
        return 0
    
    try:
        if "Z" in published_at:
            pub_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        else:
            pub_dt = datetime.fromisoformat(published_at)
        
        now = datetime.now(timezone.utc)
        delta = now - pub_dt.astimezone(timezone.utc)
        return max(1, delta.days)  # Minimum 1 day to avoid division by zero
    except Exception:
        return 1


def run_silver_transformation_v2() -> None:
    """
    Enhanced silver transformation V2 with:
    - Proper timestamp parsing (clean format: YYYY-MM-DD HH:MM:SS)
    - Extended channel metadata (custom URL, thumbnails, branding, topic categories)
    - Video performance metrics (engagement rate, views per day, duration in seconds)
    - Trending video support
    - Tags extraction for video analysis
    """
    SILVER_V2_DIR.mkdir(parents=True, exist_ok=True)

    channel_rows: list[dict] = []
    video_rows: list[dict] = []
    trending_rows: list[dict] = []

    json_files = list(BRONZE_V2_DIR.glob("**/*.json"))
    if not json_files:
        print("[SILVER-V2] No JSON files found in bronze_data_v2.")
        return

    print(f"[SILVER-V2] Processing {len(json_files)} JSON files...")

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[SILVER-V2] Failed to read {path}: {e}")
            continue

        items = raw.get("items", [])
        ingest_ts_raw = raw.get("ingestion_timestamp", "")
        ingest_ts_clean = _parse_iso_timestamp(ingest_ts_raw)
        data_type = raw.get("data_type", "unknown")

        for item in items:
            kind = item.get("kind", "")

            # --- CHANNELS ---
            if "channel" in kind:
                stats = item.get("statistics", {})
                snippet = item.get("snippet", {})
                branding = item.get("brandingSettings", {}).get("channel", {})
                topics = item.get("topicDetails", {})
                
                channel_rows.append({
                    "channel_id": item.get("id"),
                    "channel_title": snippet.get("title"),
                    "custom_url": snippet.get("customUrl"),
                    "description": snippet.get("description"),
                    "country": snippet.get("country"),
                    "published_at": _parse_iso_timestamp(snippet.get("publishedAt")),
                    "thumbnail_url": _extract_thumbnail_url(snippet.get("thumbnails", {})),
                    "subscriber_count": _safe_int(stats.get("subscriberCount")),
                    "view_count": _safe_int(stats.get("viewCount")),
                    "video_count": _safe_int(stats.get("videoCount")),
                    "keywords": branding.get("keywords"),
                    "topic_categories": ",".join(topics.get("topicCategories", [])),
                    "ingestion_timestamp": ingest_ts_clean,
                })

            # --- VIDEOS (from channel uploads or trending) ---
            elif "video" in kind:
                stats = item.get("statistics", {})
                snippet = item.get("snippet", {})
                details = item.get("contentDetails", {})
                topics = item.get("topicDetails", {})
                
                view_count = _safe_int(stats.get("viewCount"))
                like_count = _safe_int(stats.get("likeCount"))
                comment_count = _safe_int(stats.get("commentCount"))
                duration_seconds = _parse_duration_to_seconds(details.get("duration"))
                published_at = snippet.get("publishedAt")
                days_since_publish = _calculate_days_since_publish(published_at)
                
                # Calculate performance metrics
                engagement_rate = _calculate_engagement_rate(like_count, comment_count, view_count)
                views_per_day = view_count / days_since_publish if days_since_publish > 0 else 0
                
                # Extract tags (useful for trend analysis)
                tags = snippet.get("tags", [])
                tags_str = ",".join(tags[:10]) if tags else None  # Limit to first 10 tags
                
                video_data = {
                    "video_id": item.get("id"),
                    "channel_id": snippet.get("channelId"),
                    "video_title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "published_at": _parse_iso_timestamp(published_at),
                    "thumbnail_url": _extract_thumbnail_url(snippet.get("thumbnails", {})),
                    "category_id": snippet.get("categoryId"),
                    "tags": tags_str,
                    "view_count": view_count,
                    "like_count": like_count,
                    "comment_count": comment_count,
                    "duration_seconds": duration_seconds,
                    "duration_iso": details.get("duration"),
                    "definition": details.get("definition"),
                    "caption": details.get("caption"),
                    "topic_categories": ",".join(topics.get("topicCategories", [])),
                    "engagement_rate": round(engagement_rate, 4),
                    "views_per_day": round(views_per_day, 2),
                    "days_since_publish": days_since_publish,
                    "ingestion_timestamp": ingest_ts_clean,
                }
                
                # Separate trending videos from regular channel videos
                if data_type == "trending_videos":
                    video_data["trending_region"] = raw.get("region", "US")
                    trending_rows.append(video_data)
                else:
                    video_rows.append(video_data)

    # --- Save to Parquet ---
    if channel_rows:
        df_channels = pl.DataFrame(channel_rows)
        output_path = SILVER_V2_DIR / "channels_v2.parquet"
        df_channels.write_parquet(output_path)
        print(f"[SILVER-V2] ✓ Saved {len(channel_rows)} channel rows to {output_path}")
    else:
        print("[SILVER-V2] No channel rows to save.")

    if video_rows:
        df_videos = pl.DataFrame(video_rows)
        output_path = SILVER_V2_DIR / "videos_v2.parquet"
        df_videos.write_parquet(output_path)
        print(f"[SILVER-V2] ✓ Saved {len(video_rows)} video rows to {output_path}")
    else:
        print("[SILVER-V2] No video rows to save.")

    if trending_rows:
        df_trending = pl.DataFrame(trending_rows)
        output_path = SILVER_V2_DIR / "trending_videos_v2.parquet"
        df_trending.write_parquet(output_path)
        print(f"[SILVER-V2] ✓ Saved {len(trending_rows)} trending video rows to {output_path}")
    else:
        print("[SILVER-V2] No trending video rows to save.")


if __name__ == "__main__":
    run_silver_transformation_v2()
