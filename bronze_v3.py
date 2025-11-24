import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

BRONZE_V3_DIR = Path("bronze_data_v3")


class YouTubeBronzeIngestionV3:
    """
    YouTube ingestion V3 for Attention 500 analytics:
    - Comprehensive channel metrics (including hidden gems like customUrl, thumbnails)
    - Smart video strategy: Top performers + trending videos
    - Proper timezone-aware timestamps
    - Regional trending support
    """

    BASE_URL = "https://www.googleapis.com/youtube/v3"

    def __init__(
        self,
        api_key: str,
        output_dir: str | Path,
        top_videos_per_channel: int = 50,
        include_trending: bool = True,
        trending_region: str = "US",
        trending_category: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_videos_per_channel = top_videos_per_channel
        self.include_trending = include_trending
        self.trending_region = trending_region
        self.trending_category = trending_category

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{endpoint}"
        params = {**params, "key": self.api_key}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _fetch_channel_details(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch comprehensive channel data including:
        - snippet: title, description, customUrl, publishedAt, thumbnails, country
        - statistics: viewCount, subscriberCount, videoCount
        - contentDetails: relatedPlaylists
        - brandingSettings: channel keywords, banner image
        - topicDetails: topic categories
        """
        data = self._get(
            "channels",
            {
                "part": "snippet,statistics,contentDetails,brandingSettings,topicDetails",
                "id": channel_id,
            },
        )
        items = data.get("items", [])
        if not items:
            return None
        return items[0]

    def _fetch_video_ids_from_uploads_playlist(
        self, uploads_playlist_id: str, max_results: int
    ) -> List[str]:
        """Get most recent videos from uploads playlist"""
        video_ids: List[str] = []
        next_page_token: Optional[str] = None

        while len(video_ids) < max_results:
            params = {
                "part": "contentDetails",
                "playlistId": uploads_playlist_id,
                "maxResults": min(50, max_results - len(video_ids)),
            }
            if next_page_token:
                params["pageToken"] = next_page_token

            data = self._get("playlistItems", params)
            for item in data.get("items", []):
                video_id = item.get("contentDetails", {}).get("videoId")
                if video_id:
                    video_ids.append(video_id)

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

        return video_ids

    def _fetch_trending_videos(
        self, region: str = "US", max_results: int = 50, category_id: Optional[str] = None
    ) -> List[str]:
        """
        Fetch trending video IDs for a region.
        Common category IDs:
        - 10: Music
        - 17: Sports
        - 20: Gaming
        - 22: People & Blogs
        - 24: Entertainment
        - 25: News & Politics
        - 28: Science & Technology
        """
        video_ids: List[str] = []
        
        params = {
            "part": "id",
            "chart": "mostPopular",
            "regionCode": region,
            "maxResults": min(50, max_results),
        }
        
        if category_id:
            params["videoCategoryId"] = category_id
        
        data = self._get("videos", params)
        
        for item in data.get("items", []):
            video_id = item.get("id")
            if video_id:
                video_ids.append(video_id)
        
        return video_ids

    def _fetch_videos_details(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch comprehensive video details:
        - snippet: title, description, channelId, publishedAt, tags, categoryId, thumbnails
        - statistics: viewCount, likeCount, commentCount
        - contentDetails: duration, definition, caption
        - topicDetails: topic categories (if available)
        """
        all_video_items: List[Dict[str, Any]] = []

        # YouTube API allows up to 50 IDs per videos.list call
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i : i + 50]
            data = self._get(
                "videos",
                {
                    "part": "snippet,statistics,contentDetails,topicDetails",
                    "id": ",".join(batch),
                },
            )
            all_video_items.extend(data.get("items", []))

        return all_video_items

    def ingest_channel_data(self, channel_id: str) -> Path:
        """
        Fetch channel + top performing videos and save to JSON.
        Returns the path to the saved JSON.
        """
        # Get current timestamp in ISO 8601 format with UTC timezone
        ingest_timestamp = datetime.now(timezone.utc).isoformat()
        
        channel_item = self._fetch_channel_details(channel_id)
        if channel_item is None:
            raise RuntimeError(f"No data returned for channel_id={channel_id}")

        uploads_playlist_id = (
            channel_item.get("contentDetails", {})
            .get("relatedPlaylists", {})
            .get("uploads")
        )

        video_items: List[Dict[str, Any]] = []
        if uploads_playlist_id:
            video_ids = self._fetch_video_ids_from_uploads_playlist(
                uploads_playlist_id, self.top_videos_per_channel
            )
            if video_ids:
                video_items = self._fetch_videos_details(video_ids)

        payload: Dict[str, Any] = {
            "channel_id": channel_id,
            "ingestion_timestamp": ingest_timestamp,
            "data_type": "channel_top_videos",
            "items": [channel_item] + video_items,
        }

        # Use a cleaner timestamp format for filename
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"channel_{channel_id}_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return out_path

    def ingest_trending_videos(self) -> Optional[Path]:
        """
        Fetch trending videos and save to JSON.
        Returns the path to the saved JSON.
        """
        if not self.include_trending:
            return None
            
        ingest_timestamp = datetime.now(timezone.utc).isoformat()
        
        video_ids = self._fetch_trending_videos(
            region=self.trending_region,
            max_results=50,
            category_id=self.trending_category
        )
        
        if not video_ids:
            print("[BRONZE-V3] No trending videos found.")
            return None
        
        video_items = self._fetch_videos_details(video_ids)
        
        payload: Dict[str, Any] = {
            "ingestion_timestamp": ingest_timestamp,
            "data_type": "trending_videos",
            "region": self.trending_region,
            "category_id": self.trending_category,
            "items": video_items,
        }
        
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"trending_{self.trending_region}_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        
        return out_path


def get_channel_ids_v3() -> list[str]:
    """
    Get channel IDs from YOUTUBE_CHANNEL_IDS environment variable (comma-separated).
    This allows you to manage the list in GitHub Secrets.
    """
    env_val = os.getenv("YOUTUBE_CHANNEL_IDS")
    if not env_val:
        raise RuntimeError(
            "YOUTUBE_CHANNEL_IDS not set. Add comma-separated channel IDs to .env or GitHub Secrets."
        )
    
    channel_ids = [c.strip() for c in env_val.split(",") if c.strip()]
    
    if not channel_ids:
        raise RuntimeError("YOUTUBE_CHANNEL_IDS is empty. Please add channel IDs.")
    
    return channel_ids


def run_bronze_ingestion_v3(channel_ids: list[str] | None = None, include_trending: bool = True) -> None:
    """
    Bronze ingestion V3 for Attention 500 analytics.
    Dumps to bronze_data_v3 directory to preserve V2 progress.
    """
    load_dotenv()

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY not set. Add it to .env or GitHub Secrets.")

    BRONZE_V3_DIR.mkdir(parents=True, exist_ok=True)

    bronze = YouTubeBronzeIngestionV3(
        api_key=api_key,
        output_dir=str(BRONZE_V3_DIR),
        top_videos_per_channel=50,
        include_trending=include_trending,
        trending_region="US",  # Can be moved to env var if needed
    )

    if channel_ids is None:
        channel_ids = get_channel_ids_v3()

    print(f"[BRONZE-V3] Starting ingestion for {len(channel_ids)} channels...")
    all_results: list[Path] = []

    # Ingest channel data
    for idx, channel_id in enumerate(channel_ids, 1):
        print(f"[BRONZE-V3] ({idx}/{len(channel_ids)}) Ingesting channel {channel_id}...")
        try:
            path = bronze.ingest_channel_data(channel_id)
            all_results.append(path)
            print(f"[BRONZE-V3] ✓ Saved channel data to {path.name}")
        except Exception as e:
            print(f"[BRONZE-V3] ✗ Error ingesting {channel_id}: {e}")

    # Ingest trending videos
    if include_trending:
        print(f"[BRONZE-V3] Ingesting trending videos (region: US)...")
        try:
            trending_path = bronze.ingest_trending_videos()
            if trending_path:
                all_results.append(trending_path)
                print(f"[BRONZE-V3] ✓ Saved trending data to {trending_path.name}")
        except Exception as e:
            print(f"[BRONZE-V3] ✗ Error ingesting trending videos: {e}")

    print(f"\n[BRONZE-V3] ✅ Ingestion complete! Processed {len(all_results)} files.")


if __name__ == "__main__":
    run_bronze_ingestion_v3()
