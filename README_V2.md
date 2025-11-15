# YouTube Pipeline V2 - Enhanced Social Blade-Style Analytics

## 🚀 What's New in V2?

### 1. **Enhanced Channel Data Capture**
Your original pipeline was missing several valuable channel attributes. V2 now captures:

**Original (V1):**
- channel_id, channel_title, description, country, published_at
- subscriber_count, view_count, video_count

**Enhanced (V2):**
- All V1 fields PLUS:
- `custom_url` - Channel's custom URL (e.g., @MrBeast)
- `thumbnail_url` - High-quality channel avatar
- `keywords` - Channel keywords for SEO analysis
- `topic_categories` - YouTube's topic classifications
- Clean timestamps: `2025-11-15 02:13:05` instead of `2025-11-15T02:13:05.493325`

### 2. **Smarter Video Strategy**

**Problem with V1:** Just grabbing the latest 50 uploads doesn't give you the "hot 100" or performance insights.

**V2 Solution:**
- **Top Videos per Channel**: Still captures recent uploads (50 per channel)
- **Trending Videos**: NEW - captures YouTube's trending videos (Top 50 in US)
- **Performance Metrics**: 
  - `engagement_rate` = (likes + comments) / views * 100
  - `views_per_day` = total views / days since publish
  - `days_since_publish` = age of video
  - `duration_seconds` = video length (parsed from ISO duration)
  
**Video Fields Added:**
- thumbnail_url
- category_id
- tags (first 10 tags)
- definition (HD/SD)
- caption (availability)
- topic_categories
- engagement_rate
- views_per_day

### 3. **Increased Frequency: 6 Hours → 1 Hour**
- Airflow DAG now runs `@hourly` instead of every 6 hours
- Better for tracking viral videos and rapid growth patterns

### 4. **Clean Timestamp Processing**
- **Before**: `2025-11-15T02:13:05.493325` (unworkable)
- **After**: `2025-11-15 02:13:05` (SQL-friendly, human-readable)
- Properly timezone-aware (all UTC)

### 5. **Separate Tables - No Data Loss**
Your existing pipeline continues running with tables:
- `channels_log`
- `videos_log`

V2 creates NEW tables:
- `channels_log_v2` (enhanced fields)
- `videos_log_v2` (performance metrics)
- `trending_videos_log_v2` (NEW - trending content)

---

## 📋 Prerequisites

1. **Environment Variables** (Add to `.env` or GitHub Secrets):
```bash
YOUTUBE_API_KEY=your_youtube_api_key_here
NEON_DATABASE_URL=postgresql://user:pass@host/database
YOUTUBE_CHANNEL_IDS=UCbCmjCuTUZos6Inko4u57UQ,UCJplp5SjeGSdVdwsfb9Q7lQ,UC4TTx9XmK2KYCtbiN_knQvA
```

2. **Channel IDs from Your Screenshot** (20+ channels):
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

## 🎯 Usage

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

### Option 3: Airflow DAG (Production)
1. Copy `pipeline_v2.py` to your Airflow `dags/` folder
2. Copy `bronze_v2.py`, `silver_v2.py`, `gold_v2.py` to `project_scripts/`
3. DAG will run hourly automatically

---

## 📊 Database Schema

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

## 📈 Analytics Queries You Can Now Run

### 1. Top Performers by Engagement
```sql
SELECT 
    video_title,
    channel_id,
    engagement_rate,
    views_per_day
FROM videos_log_v2
ORDER BY engagement_rate DESC
LIMIT 10;
```

### 2. Channel Growth Tracking
```sql
SELECT 
    channel_title,
    ingestion_timestamp,
    subscriber_count,
    view_count
FROM channels_log_v2
WHERE channel_id = 'UCX6OQ3DkcsbYNE6H8uQQuVA'  -- MrBeast
ORDER BY ingestion_timestamp DESC;
```

### 3. Trending Videos Analysis
```sql
SELECT 
    video_title,
    channel_id,
    views_per_day,
    engagement_rate,
    tags
FROM trending_videos_log_v2
ORDER BY views_per_day DESC
LIMIT 20;
```

### 4. Video Duration vs Performance
```sql
SELECT 
    CASE 
        WHEN duration_seconds < 60 THEN 'Shorts (<1min)'
        WHEN duration_seconds < 300 THEN 'Short (1-5min)'
        WHEN duration_seconds < 600 THEN 'Medium (5-10min)'
        ELSE 'Long (10+min)'
    END as duration_category,
    AVG(engagement_rate) as avg_engagement,
    AVG(views_per_day) as avg_views_per_day
FROM videos_log_v2
GROUP BY duration_category;
```

---

## 🔧 GitHub Actions Setup

Add to `.github/workflows/youtube-pipeline.yml`:

```yaml
name: YouTube Pipeline V2

on:
  schedule:
    - cron: '0 * * * *'  # Every hour
  workflow_dispatch:      # Manual trigger

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run Pipeline V2
        env:
          YOUTUBE_API_KEY: ${{ secrets.YOUTUBE_API_KEY }}
          NEON_DATABASE_URL: ${{ secrets.NEON_DATABASE_URL }}
          YOUTUBE_CHANNEL_IDS: ${{ secrets.YOUTUBE_CHANNEL_IDS }}
        run: |
          python run_pipeline_v2.py
```

---

## 🎨 Building Your Mini Social Blade

With this enhanced pipeline, you can now build:

1. **Channel Dashboard**: Track subscriber growth, view count trends
2. **Video Performance**: Identify viral videos, engagement patterns
3. **Trending Analysis**: See what's hot in your niche
4. **Competitive Analysis**: Compare channels side-by-side
5. **Content Strategy**: Analyze tags, duration, topics that work

---

## 🚦 Next Steps

1. **Add to GitHub Secrets**:
   - `YOUTUBE_API_KEY`
   - `NEON_DATABASE_URL`
   - `YOUTUBE_CHANNEL_IDS` (comma-separated)

2. **Test Locally**:
   ```bash
   python run_pipeline_v2.py
   ```

3. **Verify Tables**:
   Check your Neon database for `channels_log_v2`, `videos_log_v2`, `trending_videos_log_v2`

4. **Deploy to Airflow** (if using):
   Copy files to your Airflow instance

5. **Build Analytics Dashboard**:
   Connect Streamlit/Tableau/Metabase to your Neon database

---

## 📝 File Structure

```
project/
├── bronze_v2.py              # Enhanced API ingestion
├── silver_v2.py              # Enhanced transformation
├── gold_v2.py                # Database load (V2 tables)
├── run_pipeline_v2.py        # One-button runner
├── pipeline_v2.py            # Airflow DAG (hourly)
├── requirements.txt          # Dependencies
├── .env                      # Local env vars
├── bronze_data_v2/           # Raw JSON files
├── silver_data_v2/           # Parquet files
└── README_V2.md              # This file
```

---

## ⚡ Performance Tips

1. **API Quota**: YouTube API has a 10,000 units/day quota
   - Each channel fetch ≈ 100-150 units
   - 20 channels/hour = ~3,000 units/hour
   - Monitor your quota at: https://console.cloud.google.com/apis/dashboard

2. **Database Indexing**:
   ```sql
   CREATE INDEX idx_channel_id ON channels_log_v2(channel_id);
   CREATE INDEX idx_video_channel ON videos_log_v2(channel_id);
   CREATE INDEX idx_ingestion_ts ON videos_log_v2(ingestion_timestamp);
   ```

3. **Data Retention**: Consider purging old records or archiving after 90 days

---

**Happy Building! 🚀**
