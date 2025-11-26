import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import polars as pl

BRONZE_V3_DIR = Path("bronze_data_v3")
SILVER_V3_DIR = Path("silver_data_v3")


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


def _calculate_milestone_metrics(view_count: int, views_per_day: float) -> Dict[str, Any]:
    """
    V3 MILESTONE TRACKING SYSTEM
    
    Track progress toward major milestones: 10M, 25M, 50M, 100M, 250M, 500M, 1B
    Approaching threshold: within 5% of next milestone
    
    Returns dict with milestone data optimized for ML models
    """
    
    # Define milestone tiers (in ascending order)
    milestones = [
        10_000_000,   # 10M
        25_000_000,   # 25M
        50_000_000,   # 50M
        100_000_000,  # 100M
        250_000_000,  # 250M
        500_000_000,  # 500M
        1_000_000_000 # 1B
    ]
    
    # Find next milestone
    next_milestone = None
    for m in milestones:
        if view_count < m:
            next_milestone = m
            break
    
    # If already passed 1B, next milestones are 2B, 3B, etc.
    if next_milestone is None:
        billions_passed = view_count // 1_000_000_000
        next_milestone = (billions_passed + 1) * 1_000_000_000
    
    # Calculate metrics
    views_to_next_milestone = next_milestone - view_count
    
    # Estimate days to milestone (None if no momentum)
    days_to_milestone = (views_to_next_milestone / views_per_day) if views_per_day > 0 else None
    
    # Calculate milestone progress percentage
    # Find the previous milestone to calculate progress properly
    prev_milestone = 0
    for m in milestones:
        if m < next_milestone:
            prev_milestone = m
        else:
            break
    
    # If next milestone is beyond 1B (like 2B, 3B), prev is the last billion mark
    if next_milestone > 1_000_000_000 and prev_milestone == 1_000_000_000:
        prev_milestone = next_milestone - 1_000_000_000
    
    milestone_range = next_milestone - prev_milestone
    progress_in_range = view_count - prev_milestone
    milestone_progress_pct = (progress_in_range / milestone_range * 100) if milestone_range > 0 else 0.0
    
    # Flag if approaching milestone (within 5% of next milestone)
    approaching_threshold = next_milestone * 0.05
    is_approaching_milestone = (views_to_next_milestone <= approaching_threshold) and (views_per_day > 0)
    
    # Calculate milestone tier for categorization
    if view_count >= 1_000_000_000:
        milestone_tier = "1B+"
    elif view_count >= 500_000_000:
        milestone_tier = "500M-1B"
    elif view_count >= 250_000_000:
        milestone_tier = "250M-500M"
    elif view_count >= 100_000_000:
        milestone_tier = "100M-250M"
    elif view_count >= 50_000_000:
        milestone_tier = "50M-100M"
    elif view_count >= 25_000_000:
        milestone_tier = "25M-50M"
    elif view_count >= 10_000_000:
        milestone_tier = "10M-25M"
    else:
        milestone_tier = "under_10M"
    
    return {
        "next_milestone": next_milestone,
        "views_to_next_milestone": views_to_next_milestone,
        "days_to_milestone": days_to_milestone,
        "is_approaching_milestone": is_approaching_milestone,
        "milestone_tier": milestone_tier,
        "milestone_progress_pct": round(milestone_progress_pct, 2),
    }


def run_silver_transformation_v3() -> None:
    """
    V3 Silver Transformation - Attention 500 Analytics
    
    ENHANCEMENTS FROM V2:
    - Joseph's ratio metrics (like_view_ratio, comment_view_ratio, like_comment_ratio)
    - Joseph's attention segments (billionaires_watch, milestones_watch, highly_viral)
    
    NEW IN V3:
    - Advanced milestone tracking system (10M, 25M, 50M, 100M, 250M, 500M, 1B)
    - Progress tracking with 5% approaching threshold
    - Days to milestone estimates for ML models
    - Milestone tier categorization
    - Channel metadata joined to video records (channel_title, custom_url)
    """
    SILVER_V3_DIR.mkdir(parents=True, exist_ok=True)

    channel_rows: list[dict] = []
    video_rows: list[dict] = []
    trending_rows: list[dict] = []

    json_files = list(BRONZE_V3_DIR.glob("**/*.json"))
    if not json_files:
        print("[SILVER-V3] No JSON files found in bronze_data_v3.")
        return

    print(f"[SILVER-V3] Processing {len(json_files)} JSON files...")

    # First pass: collect all channel metadata into a lookup dict
    channel_metadata = {}  # channel_id -> {channel_title, custom_url}

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[SILVER-V3] Failed to read {path}: {e}")
            continue

        items = raw.get("items", [])
        for item in items:
            kind = item.get("kind", "")
            
            # Build channel metadata lookup
            if "channel" in kind:
                channel_id = item.get("id")
                snippet = item.get("snippet", {})
                channel_metadata[channel_id] = {
                    "channel_title": snippet.get("title"),
                    "custom_url": snippet.get("customUrl"),
                }

    # Second pass: process all data with channel metadata available
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[SILVER-V3] Failed to read {path}: {e}")
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
                
                channel_id = snippet.get("channelId")
                
                # Get channel metadata from lookup
                channel_info = channel_metadata.get(channel_id, {})
                channel_title = channel_info.get("channel_title", snippet.get("channelTitle"))  # Fallback to snippet
                custom_url = channel_info.get("custom_url")
                
                view_count = _safe_int(stats.get("viewCount"))
                like_count = _safe_int(stats.get("likeCount"))
                comment_count = _safe_int(stats.get("commentCount"))
                duration_seconds = _parse_duration_to_seconds(details.get("duration"))
                published_at = snippet.get("publishedAt")
                days_since_publish = _calculate_days_since_publish(published_at)
                
                # Calculate performance metrics
                engagement_rate = _calculate_engagement_rate(like_count, comment_count, view_count)
                views_per_day = view_count / days_since_publish if days_since_publish > 0 else 0
                
                # V2 METRICS (Joseph's additions)
                # Like-to-view ratio (how positive the audience response is)
                like_view_ratio = (like_count / view_count) if view_count > 0 else 0.0

                # Comment-to-view ratio (indicates controversy or high engagement)
                comment_view_ratio = (comment_count / view_count) if view_count > 0 else 0.0

                # Like-to-comment ratio (high value means positive sentiment)
                like_comment_ratio = (like_count / comment_count) if comment_count > 0 else 0.0

                # --- Attention 500 index: segmentation flags (Joseph's) ---

                # Billionaires watch: videos over 1 billion views
                is_billionaires_watch = view_count >= 1_000_000_000

                # Milestones watch: videos near 100-million milestones
                # Define "near" as between 90M and 110M views
                is_milestones_watch = 90_000_000 <= view_count < 110_000_000

                # Highly viral videos: very strong momentum or engagement
                is_highly_viral = (views_per_day >= 1_000_000) or (engagement_rate >= 5.0)

                # Build a comma-separated segment label (multi-label)
                segments = []
                if is_billionaires_watch:
                    segments.append("billionaires_watch")
                if is_milestones_watch:
                    segments.append("milestones_watch")
                if is_highly_viral:
                    segments.append("highly_viral")

                attention_segment = ",".join(segments) if segments else "other"
                
                # V3 NEW: MILESTONE TRACKING SYSTEM
                milestone_data = _calculate_milestone_metrics(view_count, views_per_day)
                
                # Extract tags (useful for trend analysis)
                tags = snippet.get("tags", [])
                tags_str = ",".join(tags[:10]) if tags else None  # Limit to first 10 tags
                
                video_data = {
                    "video_id": item.get("id"),
                    "channel_id": channel_id,
                    "channel_title": channel_title,  # NEW IN V3
                    "custom_url": custom_url,  # NEW IN V3
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
                    
                    # V2 metrics (Joseph's)
                    "is_billionaires_watch": is_billionaires_watch,
                    "is_milestones_watch": is_milestones_watch,
                    "is_highly_viral": is_highly_viral,    
                    "attention_segment": attention_segment,
                    "like_view_ratio": round(like_view_ratio, 6),
                    "comment_view_ratio": round(comment_view_ratio, 6),
                    "like_comment_ratio": round(like_comment_ratio, 4),
                    
                    # V3 NEW: Milestone tracking
                    "next_milestone": milestone_data["next_milestone"],
                    "views_to_next_milestone": milestone_data["views_to_next_milestone"],
                    "days_to_milestone": milestone_data["days_to_milestone"],
                    "is_approaching_milestone": milestone_data["is_approaching_milestone"],
                    "milestone_tier": milestone_data["milestone_tier"],
                    "milestone_progress_pct": milestone_data["milestone_progress_pct"],
                    
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
        output_path = SILVER_V3_DIR / "channels_v3.parquet"
        df_channels.write_parquet(output_path)
        print(f"[SILVER-V3] ✓ Saved {len(channel_rows)} channel rows to {output_path}")
    else:
        print("[SILVER-V3] No channel rows to save.")

    if video_rows:
        df_videos = pl.DataFrame(video_rows)
        output_path = SILVER_V3_DIR / "videos_v3.parquet"
        df_videos.write_parquet(output_path)
        print(f"[SILVER-V3] ✓ Saved {len(video_rows)} video rows to {output_path}")
    else:
        print("[SILVER-V3] No video rows to save.")

    if trending_rows:
        df_trending = pl.DataFrame(trending_rows)
        output_path = SILVER_V3_DIR / "trending_videos_v3.parquet"
        df_trending.write_parquet(output_path)
        print(f"[SILVER-V3] ✓ Saved {len(trending_rows)} trending video rows to {output_path}")
    else:
        print("[SILVER-V3] No trending video rows to save.")


if __name__ == "__main__":
    run_silver_transformation_v3()
