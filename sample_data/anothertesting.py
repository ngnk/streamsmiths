import os
import csv
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
API_KEY = os.environ["YOUTUBE_API_KEY"]
yt = build("youtube", "v3", developerKey=API_KEY)

MAX_VIDEOS_PER_CHANNEL = 10

def handle_to_channel_id(handle: str) -> Optional[str]:
    """
    Resolve a YouTube @handle to a channel ID.
    """
    resp = yt.channels().list(
        part="id",
        forHandle=handle.lstrip("@")
    ).execute()
    items = resp.get("items", [])
    return items[0]["id"] if items else None


def get_channel_record(channel_id: str, handle: str) -> Dict[str, Any]:
    """
    Fetch channel-level fields (one row per channel).

    Fields:
      - handle
      - id
      - title
      - country
      - viewCount
      - subscriberCount
      - videoCount
      - channel.title
      - channel.keywords
      - uploads_playlist  (useful for fetching videos)
    """
    resp = yt.channels().list(
        part="snippet,statistics,brandingSettings,contentDetails",
        id=channel_id
    ).execute()

    c = resp["items"][0]
    snippet = c.get("snippet", {})
    stats = c.get("statistics", {}) or {}
    branding_channel = (c.get("brandingSettings", {}) or {}).get("channel", {}) or {}
    content = c.get("contentDetails", {}) or {}
    uploads_playlist = (content.get("relatedPlaylists", {}) or {}).get("uploads")

    # channel.keywords is a list -> join into a single string for CSV
    keywords_list = branding_channel.get("keywords", [])
    if isinstance(keywords_list, list):
        keywords_str = ",".join(keywords_list)
    else:
        keywords_str = str(keywords_list) if keywords_list is not None else ""

    return {
        "handle": handle,
        "id": c.get("id"),
        "title": snippet.get("title"),
        "country": snippet.get("country"),
        "viewCount": stats.get("viewCount"),
        "subscriberCount": stats.get("subscriberCount"),
        "videoCount": stats.get("videoCount"),
        "channel.title": branding_channel.get("title"),
        "channel.keywords": keywords_str,
        "uploads_playlist": uploads_playlist,
    }


def get_uploads_playlist_id(channel_id: str) -> Optional[str]:
    """Get the uploads playlist ID for a channel."""
    resp = yt.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()
    items = resp.get("items", [])
    if not items:
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def get_video_ids_from_uploads(playlist_id: str,
                               max_results: int = MAX_VIDEOS_PER_CHANNEL) -> List[str]:
    """
    Get up to `max_results` video IDs from the uploads playlist.
    """
    video_ids: List[str] = []
    page_token = None

    while len(video_ids) < max_results:
        resp = yt.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=min(50, max_results - len(video_ids)),
            pageToken=page_token
        ).execute()

        for item in resp.get("items", []):
            vid = item["contentDetails"]["videoId"]
            video_ids.append(vid)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return video_ids


def chunked(seq: List[str], n: int) -> List[List[str]]:
    """Yield successive n-sized chunks from a list."""
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def get_video_metadata(video_ids: List[str]) -> List[Dict[str, Any]]:
    """
    For a list of video IDs, fetch the video-level fields from your sheet:

      - id
      - publishedAt
      - channelId
      - channelTitle
      - title
      - categoryId
      - viewCount
      - likeCount
      - dislikeCount
      - commentCount
      - duration
    """
    if not video_ids:
        return []

    resp = yt.videos().list(
        part="snippet,statistics,contentDetails",
        id=",".join(video_ids)
    ).execute()

    rows: List[Dict[str, Any]] = []
    for v in resp.get("items", []):
        sn = v.get("snippet", {}) or {}
        st = v.get("statistics", {}) or {}
        cd = v.get("contentDetails", {}) or {}

        rows.append({
            "id": v.get("id"),
            "publishedAt": sn.get("publishedAt"),
            "channelId": sn.get("channelId"),
            "channelTitle": sn.get("channelTitle"),
            "title": sn.get("title"),
            "categoryId": sn.get("categoryId"),

            "viewCount": st.get("viewCount"),
            "likeCount": st.get("likeCount"),
            "dislikeCount": st.get("dislikeCount"),
            "commentCount": st.get("commentCount"),

            "duration": cd.get("duration"),  # ISO 8601
        })

    return rows


def get_category_titles(category_ids: List[str],
                        region: str = "US") -> List[Dict[str, Any]]:
    """
    Given a list of category IDs, return rows with:
      - id
      - title

    NOTE: When specifying `id`, the YouTube API does NOT allow `regionCode`
    at the same time, so we omit regionCode here.
    """
    if not category_ids:
        return []

    unique_ids = sorted(set(cid for cid in category_ids if cid))
    rows: List[Dict[str, Any]] = []

    # videoCategories.list limit is 50 IDs per call
    for chunk in chunked(unique_ids, 50):
        resp = yt.videoCategories().list(
            part="snippet",
            id=",".join(chunk)
        ).execute()

        for c in resp.get("items", []):
            rows.append({
                "id": c.get("id"),
                "title": c.get("snippet", {}).get("title"),
            })

    # de-duplicate in case of overlap
    seen = set()
    deduped = []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            deduped.append(r)
    return deduped


# Main: dump everything to CSVs
if __name__ == "__main__":
    handles = [
        "@TaylorSwift",
        # "@AnotherHandle",
    ]

    channel_rows: List[Dict[str, Any]] = []
    video_rows: List[Dict[str, Any]] = []
    all_category_ids: List[str] = []

    for handle in handles:
        print(f"Processing handle: {handle}")
        cid = handle_to_channel_id(handle)
        if cid is None:
            print(f"  [WARN] Could not resolve channel for handle {handle}")
            continue

        # Channel info
        chan_rec = get_channel_record(cid, handle)
        channel_rows.append(chan_rec)

        # Videos for this channel
        uploads_playlist = chan_rec["uploads_playlist"]
        if not uploads_playlist:
            print(f"  [WARN] No uploads playlist for {handle}")
            continue

        vid_ids = get_video_ids_from_uploads(uploads_playlist,
                                             MAX_VIDEOS_PER_CHANNEL)
        if not vid_ids:
            print(f"  [INFO] No videos found for {handle}")
            continue

        # Fetch metadata in chunks of up to 50 IDs
        for chunk in chunked(vid_ids, 50):
            meta = get_video_metadata(chunk)
            for row in meta:
                # attach the handle so you know which channel it came from
                row["handle"] = handle
                video_rows.append(row)
                if row.get("categoryId"):
                    all_category_ids.append(row["categoryId"])

    # Category table
    category_rows = get_category_titles(all_category_ids, region="US")

    # channels.csv
    channel_fieldnames = [
        "handle",
        "id",
        "title",
        "country",
        "viewCount",
        "subscriberCount",
        "videoCount",
        "channel.title",
        "channel.keywords",
        "uploads_playlist",
    ]
    with open("channels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=channel_fieldnames)
        w.writeheader()
        w.writerows(channel_rows)
    print(f"Wrote {len(channel_rows)} rows to channels.csv")

    # videos.csv
    video_fieldnames = [
        "handle",
        "id",
        "publishedAt",
        "channelId",
        "channelTitle",
        "title",
        "categoryId",
        "viewCount",
        "likeCount",
        "dislikeCount",
        "commentCount",
        "duration",
    ]
    with open("videos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=video_fieldnames)
        w.writeheader()
        w.writerows(video_rows)
    print(f"Wrote {len(video_rows)} rows to videos.csv")

    # categories.csv
    category_fieldnames = ["id", "title"]
    with open("categories.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=category_fieldnames)
        w.writeheader()
        w.writerows(category_rows)
    print(f"Wrote {len(category_rows)} rows to categories.csv")
