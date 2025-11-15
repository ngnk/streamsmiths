---
# YouTube Pipeline V2
---

## Channels

1. **Cocomelon**
2. **LikeNastya**
3. **Ryan's World**
4. **MrBeast**
5. **PewDiePie**
6. **Sidemen**
7. **TEDx Talks**
8. **Kurzgesagt**
9. **Nick DiGiovanni**
10. **CNN**
11. **Yuya**
12. **James Charles**
13. **JuegaGerman**
14. **Markiplier**
15. **Taylor Swift**
16. **Michael Jackson**
17. **BLACKPINK**
18. **Bad Bunny**
19. **T-Series**
20. **UR Cristiano**
21. **Tibo InShape**
22. **Chloe Ting**
23. **Dude Perfect**
24. **Linus Tech Tips**

---

## Usage

### Option 1: One-Button Local Run
```bash
python run_pipeline_v2.py
```

This will:
1. Fetch data from YouTube API → `bronze_data_v2/`
2. Transform to Parquet → `silver_data_v2/`
3. Load to Neon tables → `channels_log_v2`, `videos_log_v2`, `trending_videos_log_v2`

### Option 2: Run Individual Steps
```bash
# Step 1: Bronze (API ingestion)
python bronze_v2.py

# Step 2: Silver (transformation)
python silver_v2.py

# Step 3: Gold (database load)
python gold_v2.py
```

---

## Database Schema

### `channels_log_v2`
```sql
channel_id          TEXT
channel_title       TEXT
custom_url          TEXT         -- NEW
description         TEXT
country             TEXT
published_at        TIMESTAMP    -- Clean format
thumbnail_url       TEXT         -- NEW
subscriber_count    BIGINT
view_count          BIGINT
video_count         INTEGER
keywords            TEXT         -- NEW
topic_categories    TEXT         -- NEW
ingestion_timestamp TIMESTAMP    -- Clean format
```

### `videos_log_v2`
```sql
video_id            TEXT
channel_id          TEXT
video_title         TEXT
description         TEXT
published_at        TIMESTAMP
thumbnail_url       TEXT         -- NEW
category_id         TEXT         -- NEW
tags                TEXT         -- NEW
view_count          BIGINT
like_count          INTEGER
comment_count       INTEGER
duration_seconds    INTEGER      -- NEW
duration_iso        TEXT
definition          TEXT         -- NEW (HD/SD)
caption             TEXT         -- NEW
topic_categories    TEXT         -- NEW
engagement_rate     FLOAT        -- NEW (calculated)
views_per_day       FLOAT        -- NEW (calculated)
days_since_publish  INTEGER      -- NEW
ingestion_timestamp TIMESTAMP
```

### `trending_videos_log_v2` (NEW TABLE)
Same schema as `videos_log_v2` PLUS:
```sql
trending_region     TEXT         -- 'US' by default
```

---
