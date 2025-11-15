---
# YouTube Pipeline V2
---

## Channels

```
UCbCmjCuTUZos6Inko4u57UQ  # Cocomelon
UCJplp5SjeGSdVdwsfb9Q7lQ  # LikeNastya
UC4TTx9XmK2KYCtbiN_knQvA  # Ryan's World
UCX6OQ3DkcsbYNE6H8uQQuVA  # MrBeast
UC-lHJZR3Gqxm24_Vd_AJ5Yw  # PewDiePie
UCDogdKl7t7HzQ95aEwkdMw   # Sidemen
UCsT0YIqwnpJCM-mx7-gSA4Q  # TEDx Talks
UCsxVk37bltHxD1rDPwtNM8Q  # Kurzgesagt
UCMyOj6fhvKFMjxUCp3b3gAQ  # Nick DiGiovanni
UCupvZG-5ko_eiXAupbDfxWw  # CNN
UCBNs31xyxpxAGMheg8OrgA   # Yuya
UCucot-Zp428OwkyRm2I72Q   # James Charles
UCYiGq8XF7YQD00x7wAd6zzg  # JuegaGerman
UC7_YxT-KID8kRbqZo7MyscQ  # Markiplier
UCqECaJ8Gagnn7YCbPEzWH6g  # Taylor Swift
UC50rbDVD9scpcAstz7JnQGA  # Michael Jackson
UCOmHUn--16B90oW2L6FRR3A  # Blackpink
UCmBA_wu8xGg1Qg1Of0kW13Q  # Bad Bunny
UCq-Fj5jknLsUf-MWSy4_brA  # T-Series
UCtxD0xAuNNqdXO9Wp5pGHew # UR Cristiano
UCpWaR3qNAQsGx48lQCqV0Cw  # Tibo InShape
UCCgLoMYlYoP0U56dEb06adQ  # Chloe Ting
UCRijo3ddMTht_IHyNSNXpNQ  # Dude Perfect
UCXuqSBlHAE6Xw-yeJA8sc_w  # Linus Tech Tips
```

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
