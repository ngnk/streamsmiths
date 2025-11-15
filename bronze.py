import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

BRONZE_DIR = Path("bronze_data")


class YouTubeBronzeIngestion:
    """
    Simple YouTube ingestion class:
    - Fetches channel details (snippet + statistics + contentDetails)
    - Fetches latest N uploaded videos for that channel
    - Saves combined JSON to bronze_data/<channel_id>_<timestamp>.json
    """

    BASE_URL = "https://www.googleapis.com/youtube/v3"

    def __init__(
        self,
        api_key: str,
        output_dir: str | Path,
        max_videos_per_channel: int = 50,
    ) -> None:
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_videos_per_channel = max_videos_per_channel

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{endpoint}"
        params = {**params, "key": self.api_key}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _fetch_channel_details(self, channel_id: str) -> Optional[Dict[str, Any]]:
        data = self._get(
            "channels",
            {
                "part": "snippet,statistics,contentDetails",
                "id": channel_id,
            },
        )
        items = data.get("items", [])
        if not items:
            return None
        return items[0]

    def _fetch_video_ids_from_uploads_playlist(
        self, uploads_playlist_id: str
    ) -> List[str]:
        video_ids: List[str] = []
        next_page_token: Optional[str] = None

        while True:
            params = {
                "part": "contentDetails",
                "playlistId": uploads_playlist_id,
                "maxResults": 50,
            }
            if next_page_token:
                params["pageToken"] = next_page_token

            data = self._get("playlistItems", params)
            for item in data.get("items", []):
                video_id = item.get("contentDetails", {}).get("videoId")
                if video_id:
                    video_ids.append(video_id)
                    if len(video_ids) >= self.max_videos_per_channel:
                        return video_ids

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

        return video_ids

    def _fetch_videos_details(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        all_video_items: List[Dict[str, Any]] = []

        # YouTube API allows up to 50 IDs per videos.list call
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i : i + 50]
            data = self._get(
                "videos",
                {
                    "part": "snippet,statistics,contentDetails",
                    "id": ",".join(batch),
                },
            )
            all_video_items.extend(data.get("items", []))

        return all_video_items

    def ingest_channel_data(self, channel_id: str) -> Path:
        """
        Fetch channel + latest videos and save to a single JSON file.
        Returns the path to the saved JSON.
        """
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
                uploads_playlist_id
            )
            if video_ids:
                video_items = self._fetch_videos_details(video_ids)

        payload: Dict[str, Any] = {
            "channel_id": channel_id,
            "ingestion_timestamp": datetime.utcnow().isoformat(),
            "items": [channel_item] + video_items,
        }

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_path = self.output_dir / f"{channel_id}_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return out_path


def get_channel_ids() -> list[str]:
    """
    Prefer channel IDs from env var YOUTUBE_CHANNEL_IDS (comma-separated).
    Fallback to a small hard-coded list that you can extend.
    """
    env_val = os.getenv("YOUTUBE_CHANNEL_IDS")
    if env_val:
        return [c.strip() for c in env_val.split(",") if c.strip()]

    # Fallback example list
    return [
        "UCX6OQ3DkcsbYNE6H8uQQuVA",  # MrBeast
        "UCqECaJ8Gagnn7YCbPEzWH6g",  # Ed Sheeran
    ]


def run_bronze_ingestion(channel_ids: list[str] | None = None) -> None:
    """
    Callable function for the one-button runner and for Airflow.
    """
    load_dotenv()

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY not set. Add it to .env or env vars.")

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    bronze = YouTubeBronzeIngestion(api_key=api_key, output_dir=str(BRONZE_DIR))

    if channel_ids is None:
        channel_ids = get_channel_ids()

    all_results: list[Path] = []

    for channel_id in channel_ids:
        print(f"[BRONZE] Ingesting channel {channel_id} ...")
        try:
            path = bronze.ingest_channel_data(channel_id)
            all_results.append(path)
            print(f"[BRONZE] Saved JSON to {path}")
        except Exception as e:
            print(f"[BRONZE] Error ingesting {channel_id}: {e}")

    print(f"[BRONZE] Ingestion complete. Processed {len(all_results)} channels.")


if __name__ == "__main__":
    run_bronze_ingestion()
