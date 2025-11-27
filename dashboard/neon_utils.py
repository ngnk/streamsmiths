# neon_utils.py
import pandas as pd
import psycopg2


def get_neon_conn():
    return psycopg2.connect(
        dbname="",
        user="",
        password="",
        host="",
        port="",
        sslmode="",
    )


def load_trending_v2(days: int = 7, region: str = "US") -> pd.DataFrame:
   
    conn = get_neon_conn()
    query = """
    SELECT
        t.video_id,
        t.channel_id,
        c.channel_title,
        t.video_title,
        t.view_count,
        t.like_count,
        t.comment_count,
        t.duration_seconds,
        t.engagement_rate,
        t.views_per_day,
        t.days_since_publish,
        t.ingestion_timestamp,
        t.trending_region
    FROM trending_videos_log_v2 AS t
    LEFT JOIN channels_log_v2 AS c
      ON t.channel_id = c.channel_id
    WHERE t.ingestion_timestamp::timestamptz >= NOW() - INTERVAL %s
      AND t.trending_region = %s
    """
    df = pd.read_sql(query, conn, params=(f"{days} days", region))
    conn.close()

    df["ingestion_timestamp"] = pd.to_datetime(df["ingestion_timestamp"])
    return df
