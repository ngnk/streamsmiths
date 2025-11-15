# StreamWatch - YouTube Analytics Pipeline

A medallion architecture (Bronze→Silver→Gold) data pipeline for tracking YouTube channel and video statistics.

## Architecture

- **Bronze Layer**: Raw JSON data from YouTube API
- **Silver Layer**: Transformed Parquet files with cleaned data
- **Gold Layer**: Neon PostgreSQL database for analytics
- **Dashboard**: Streamlit visualization

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:
- `YOUTUBE_API_KEY`: Get from Google Cloud Console
- `NEON_DATABASE_URL`: Your Neon PostgreSQL connection string

### 3. Get YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable YouTube Data API v3
4. Create credentials (API Key)
5. Add to `.env` file

### 4. Setup Neon Database

1. Sign up at [Neon](https://neon.tech/)
2. Create a new project
3. Copy the connection string
4. Add to `.env` as `NEON_DATABASE_URL`

The pipeline will automatically create tables on first run.

## Usage

### Run Complete Pipeline Once

```bash
python run_pipeline.py
```

This executes all three layers in sequence.

### Run Individual Layers

```bash
# Bronze: Fetch from YouTube API
python bronze.py

# Silver: Transform to Parquet
python silver.py

# Gold: Load to Neon
python gold.py
```

### Launch Dashboard

```bash
streamlit run dashboard.py
```

### Setup with Airflow (Optional)

1. Copy `pipeline.py` to your Airflow `dags/` folder
2. Copy bronze.py, silver.py, gold.py to `project_scripts/` folder
3. Ensure `.env` is accessible to Airflow
4. The DAG runs hourly by default

## Configuration

### Channel IDs

Add channels to track by setting in `.env`:

```
YOUTUBE_CHANNEL_IDS=UCX6OQ3DkcsbYNE6H8uQQuVA,UCqECaJ8Gagnn7YCbPEzWH6g
```

Or edit the fallback list in `bronze.py` (line 143).

### Max Videos Per Channel

Default: 50. Adjust in `bronze.py`:

```python
bronze = YouTubeBronzeIngestion(
    api_key=api_key,
    output_dir=str(BRONZE_DIR),
    max_videos_per_channel=100  # Change here
)
```

## Database Schema

### channels_log

| Column | Type | Description |
|--------|------|-------------|
| channel_id | TEXT | YouTube channel ID |
| channel_title | TEXT | Channel name |
| description | TEXT | Channel description |
| country | TEXT | Channel country |
| published_at | TEXT | Channel creation date |
| subscriber_count | INTEGER | Total subscribers |
| view_count | INTEGER | Total views |
| video_count | INTEGER | Total videos |
| ingest_timestamp | TEXT | When data was fetched |

### videos_log

| Column | Type | Description |
|--------|------|-------------|
| video_id | TEXT | YouTube video ID |
| channel_id | TEXT | Parent channel ID |
| video_title | TEXT | Video title |
| description | TEXT | Video description |
| published_at | TEXT | Upload date |
| view_count | INTEGER | View count |
| like_count | INTEGER | Like count |
| comment_count | INTEGER | Comment count |
| duration | TEXT | Video duration (ISO 8601) |
| ingest_timestamp | TEXT | When data was fetched |

