<div align="center">

# STREAMWATCH

### IDS 706 Fall 2025 Final Project: YouTube Analytics Platform

![V1](https://github.com/ngnk/streamsmiths/actions/workflows/workflow_v1.yml/badge.svg) ![V2](https://github.com/ngnk/streamsmiths/actions/workflows/workflow_v2.yml/badge.svg) ![V3](https://github.com/ngnk/streamsmiths/actions/workflows/workflow_v3.yml/badge.svg) ![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub%20Actions-brightgreen.svg)

**Team 6 (Streamsmiths): Tony Ngari, Can He, Matthew Fischer, Joseph Hong, Trey Chase**

</div>

## Table of Contents

- [Project Overview](#-project-overview)
- [Team](#-team)
- [Architecture](#-architecture)
- [Setup Instructions](#-setup-instructions)
- [Key Principles Implementation](#-key-principles-implementation)
- [Data Pipeline](#-data-pipeline)
- [Dashboard Features](#-dashboard-features)
- [Project Evolution](#-project-evolution)
- [Future Improvements](#-future-improvements)
- [Lessons Learned](#-lessons-learned)

---

## Project Overview

**STREAMWATCH** is a comprehensive YouTube analytics platform that provides insights into channel performance, video trends, and milestone tracking.

The platform processes data from 50-100+ YouTube channels through a **bronze-silver-gold data pipeline architecture**, delivering:
- Real-time channel and video performance tracking
- Milestone achievement monitoring
- Engagement analytics and trend identification
- Historical data analysis with interactive visualizations
- Scalable data infrastructure supporting future ML predictions

---

## Team

| Name | Role | Responsibilities |
|------|------|-----------------|
| **Tony N.** | Leader / Engineering | Pipeline architecture, database design, system integration, project coordination |
| **Joseph H.** | Engineering | Data processing workflows, API integration, model development |
| **Trey C.** | Data Science | Model development, metrics calculation |
| **Can H.** | Analytics | Dashboard development, visualization design |
| **Matthew F.** | Data Science | Model development, metrics calculation |

---

## Architecture

STREAMWATCH implements a **medallion architecture** (bronze-silver-gold) for data processing, ensuring data quality, traceability, and scalability.

### Data Flow
1. **Ingestion**: GitHub Actions trigger pipeline every 8 hours
2. **Raw Storage**: YouTube API data appended to Bronze tables
3. **Transformation**: Silver layer computes metrics and cleans data
4. **Analytics**: Gold layer creates aggregated, dashboard-ready views
5. **Visualization**: Streamlit dashboard queries latest Gold layer data

### Data Engineering
- **Pipeline**: Python
- **Database**: Neon PostgreSQL (cloud-hosted)
- **Orchestration**: GitHub Actions (YAML workflows)
- **Data Processing**: Pandas, Polars

### Dashboard
- **Framework**: Streamlit
- **Visualization**: Plotly Express, Plotly Graph Objects
- **Styling**: Custom CSS

### Data Sources
- 
[**YouTube Data API v3**][https://developers.google.com/youtube/v3]

### Development Tools
- **Version Control**: Git / GitHub
- **Environment Management**: python-dotenv
- **Database ORM**: SQLAlchemy

---

## Setup Instructions

### Prerequisites
- Python
- YouTube Data API key [Follow instructions here on how to obtain][https://www.youtube.com/watch?v=EPeDTRNKAVo]
- [Neon][https://neon.com/] database account (or desired PostgreSQL instance) _create a free account, create a project and open your psotgres instance_
- Git

**Getting a YouTube API Key:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable "YouTube Data API v3"
4. Create credentials (API key)
5. Copy the API key to `.env`

**Setting up Neon Database:**
1. Sign up at [Neon.tech](https://neon.tech)
2. Create a new project
3. Copy the connection string to `NEON_DATABASE_URL`
   
### 1. Clone the Repository
```bash
git clone https://github.com/ngnk/streamsmiths.git
cd streamwatch
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages** (`requirements.txt`):
```txt
streamlit>=1.40.0
polars>=0.20.0
plotly>=5.18.0
pandas>=2.1.0
sqlalchemy>=2.0.0
python-dotenv>=1.0.0
requests>=2.31.0
google-api-python-client>=2.100.0
psycopg2-binary>=2.9.9
```

### 3. Configure Environment Variables
Create a `.env` file in the project root:

```bash
# YouTube API Configuration
YOUTUBE_API_KEY=YOUR_YOUTUBE_API_KEY

# Database Configuration
NEON_DATABASE_URL=YOUR_DATABASE_CONNECTION_STRING

# Optional: Channel List (comma-separated channel IDs)
YOUTUBE_CHANNEL_IDS=UCupvZG-5ko_eiXAupbDfxWw,UCX6OQ3DkcsbYNE6H8uQQuVA
```

### 4. Initialize Database Tables
Run the pipeline setup script to create tables:

```bash
python scripts/setup_database.py
```

This will create:
- `channels_log_v3` (Bronze layer)
- `videos_log_v3` (Bronze layer)
- Future: Silver and Gold layer tables

### 5. Run Initial Data Ingestion
```bash
python pipeline/ingest_youtube_data.py
```

### 6. Launch Dashboard
```bash
cd dashboard
streamlit run dashboard.py
```

Access the dashboard at `http://localhost:8501`

### 7. (Optional) Set Up GitHub Actions
For automated pipeline runs:

1. Fork the repository
2. Go to **Settings → Secrets and variables → Actions**
3. Add secrets:
   - `YOUTUBE_API_KEY`
   - `NEON_DATABASE_URL`
   - `YOUTUBE_CHANNEL_IDS`
4. Enable GitHub Actions in repository settings
5. Pipeline will run automatically every hour

---

## Key Principles Implementation

### 1. Scalability
**Implementation:**
- **Horizontal Scaling**: Database-driven channel management supports 50-100+ channels (vs. 25-channel GitHub Secrets limit)
- **API Quota Efficiency**: Pipeline consumes only 77 units/run (35x under estimated 2,719), providing massive headroom
- **Time-Series Architecture**: Append-only Bronze tables support unlimited historical growth
- **Cloud Database**: Neon PostgreSQL with connection pooling handles concurrent queries

**Example:**
```python
# Scalable channel querying using SQL instead of hardcoded lists
query = """
SELECT DISTINCT channel_id FROM channels_log_v3
WHERE last_updated > NOW() - INTERVAL '7 days'
"""
# Supports 100s of channels without code changes
```

### 2. Modularity
**Implementation:**
- **Versioned Tables**: Separate V1, V2, V3 schemas preserve existing data during iteration
- **Layered Architecture**: Bronze (raw) → Silver (transformed) → Gold (analytics) separation
- **Reusable Functions**: `calculate_grade()`, `format_number()`, `load_channels()` used across dashboard
- **Independent Workflows**: Separate GitHub Actions for V1, V2, V3 pipelines

**Example:**
```python
# Modular metric calculation - easily extended
def calculate_engagement_rate(likes, comments, views):
    return ((likes + comments) / views) * 100

# Reusable across all video analytics
engagement = calculate_engagement_rate(row['like_count'], 
                                      row['comment_count'], 
                                      row['view_count'])
```

### 3. Reusability
**Implementation:**
- **Templated SQL Queries**: Parameterized queries work for channels, videos, time ranges
- **Abstracted Data Loaders**: `load_video_history()`, `load_channel_history()` functions
- **Style Components**: Reusable CSS classes (`.metric-card`, `.channel-card`, `.milestone-badge`)
- **Visualization Templates**: Plotly chart configurations used across multiple pages

**Example:**
```python
@st.cache_data(ttl=3600)
def load_history(table_name, id_column, id_value, days=30):
    """Generic time-series loader - works for channels AND videos"""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    query = f"""
    SELECT * FROM {table_name}
    WHERE {id_column} = '{id_value}'
    AND ingestion_timestamp >= '{cutoff}'
    ORDER BY ingestion_timestamp ASC
    """
    return pl.from_pandas(pd.read_sql(query, engine))
```

### 4. Observability
**Implementation:**
- **Ingestion Timestamps**: Every record tagged with `ingestion_timestamp` for lineage tracking
- **Pipeline Logging**: GitHub Actions logs capture API responses, row counts, errors
- **Version Tracking**: V1 → V2 → V3 tables preserve evolution history
- **Dashboard Metrics**: Real-time counts of channels, videos, Billionaires Club members

**Example:**
```python
# Full audit trail in database
INSERT INTO videos_log_v3 (
    video_id, 
    view_count, 
    ingestion_timestamp  -- Tracks WHEN data was captured
)
VALUES ('xyz123', 1500000, '2025-01-15 14:30:00')

# Query shows data freshness
SELECT MAX(ingestion_timestamp) as last_update 
FROM videos_log_v3
-- Result: "2025-01-15 14:30:00" (data is 2 hours old)
```

### 5. Data Governance
**Implementation:**
- **Schema Versioning**: V1, V2, V3 tables document pipeline evolution
- **Immutable Bronze Layer**: Raw API responses never modified (append-only)
- **Data Lineage**: Clear transformation path: Bronze → Silver → Gold
- **Quality Validation**: Timestamp formatting, duplicate detection, null handling

**Example:**
```python
# Silver layer transformation with quality checks
def transform_silver(bronze_data):
    # Normalize timestamps (governance rule)
    bronze_data['ingestion_timestamp'] = pd.to_datetime(
        bronze_data['ingestion_timestamp']
    ).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Deduplicate (data quality rule)
    silver_data = bronze_data.drop_duplicates(
        subset=['video_id', 'ingestion_timestamp']
    )
    
    # Validate required fields
    assert silver_data['video_id'].notna().all(), "Missing video IDs"
    
    return silver_data
```

### 6. Reliability
**Implementation:**
- **Connection Pooling**: SQLAlchemy `pool_pre_ping=True` prevents stale connections
- **Error Recovery**: Try-catch blocks in API calls with graceful degradation
- **Scheduled Automation**: GitHub Actions 8-hour cron ensures consistent data freshness
- **Caching Strategy**: Streamlit `@st.cache_data(ttl=3600)` reduces database load

**Example:**
```python
@st.cache_resource
def get_db_connection():
    """Auto-recovering database connection"""
    return create_engine(
        db_uri, 
        pool_pre_ping=True,      # Verify connection before using
        pool_recycle=1800        # Recycle connections every 30min
    )

# Pipeline handles API failures gracefully
try:
    response = youtube.videos().list(id=video_id, part='statistics').execute()
except HttpError as e:
    logging.error(f"API error for {video_id}: {e}")
    continue  # Skip this video, process others
```

### 7. Efficiency
**Implementation:**
- **API Quota Optimization**: Batch requests, selective field retrieval (`part='snippet,statistics'`)
- **Query Optimization**: `DISTINCT ON` for latest records, indexed `ingestion_timestamp`
- **Dashboard Caching**: 1-hour TTL prevents redundant database queries
- **Selective Data Loading**: Only fetch 20 videos per page, 30-day history windows

**Example:**
```python
# Efficient: Get latest record per video in ONE query
query = """
SELECT DISTINCT ON (video_id) *
FROM videos_log_v3
ORDER BY video_id, ingestion_timestamp DESC
"""
# vs. Inefficient: Query all records, filter in Python (100x slower)

# API efficiency: Batch 50 video IDs per request (max allowed)
video_ids = ','.join(video_list[:50])
response = youtube.videos().list(id=video_ids, part='statistics')
# vs. 50 separate API calls (50x quota usage)
```

### 8. Security
**Implementation:**
- **Secret Management**: API keys stored in GitHub Secrets, never committed to Git
- **Environment Variables**: `.env` file in `.gitignore`, `python-dotenv` for local dev
- **Database Encryption**: Neon provides SSL/TLS connections by default
- **No Hardcoded Credentials**: All sensitive data externalized to environment config

**Example:**
```python
# CORRECT: Secure API key handling
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("YOUTUBE_API_KEY")  # Reads from .env or environment
youtube = build('youtube', 'v3', developerKey=api_key)

# WRONG: Hardcoded credentials (never do this!)
# api_key = "AIzaSyC_1234567890abcdefg"  # EXPOSED IN GIT HISTORY!
```

**.gitignore** includes:
```
.env
*.env
secrets/
credentials.json
```

---

## Data Pipeline

### Bronze Layer (Raw Ingestion)
**Tables:**
- `channels_log_v3`: Raw channel metadata from YouTube API
- `videos_log_v3`: Raw video statistics from YouTube API

**Key Fields:**
- `channel_id` / `video_id`: Primary identifiers
- `ingestion_timestamp`: When data was captured
- `subscriber_count`, `view_count`, `video_count`: Raw metrics
- `like_count`, `comment_count`: Engagement data

### Silver Layer (Transformation)
**Computed Metrics:**
```python
engagement_rate = ((like_count + comment_count) / view_count) * 100
views_per_day = view_count / days_since_publish
like_view_ratio = like_count / view_count
comment_view_ratio = comment_count / view_count
```

**Milestone Logic:**
```python
milestones = [1_000_000_000, 500_000_000, 250_000_000, 100_000_000, 
              50_000_000, 25_000_000, 10_000_000]

next_milestone = min([m for m in milestones if m > view_count])
milestone_progress_pct = (view_count / next_milestone) * 100
is_approaching_milestone = milestone_progress_pct >= 95
```

### Gold Layer (Analytics)
**Aggregations:**
- Top channels by subscriber count
- Top videos by view count
- Billionaires Club (1B+ views)
- Milestone achievers (crossed thresholds this month)
- Average engagement rates by category

### Orchestration
**GitHub Actions Workflow** (`.github/workflows/pipeline_v3.yml`):


---

## Dashboard Features

### Home Page
- **STREAMWATCH Header**: Gradient-styled branding
- **7 Key Metrics**: Channels, Videos, Billionaires Club, Viral Videos, Total Subs, Total Views, Avg Engagement
- **Top Channels**: Leaderboard with grades (A++, A+, etc.)
- **Recent Videos**: Latest uploads across all channels

### Channel Leaderboard
- **Grading System**: 
  - `A++`: 50M+ subs, 10B+ views
  - `A+`: 10M+ subs, 1B+ views
  - `A`: 1M+ subs, 100M+ views
  - `B+`: 100K+ subs, 10M+ views
  - `B/C`: Below thresholds
- **Sortable Metrics**: Subscribers, views, engagement, video count
- **Drill-Down**: Click channel → View all videos

### Video Explorer
- **Filters**: All Videos, Billionaires Watch (1B+), Approaching Milestone, Highly Viral
- **Milestone Tiers**: 1B+, 500M-1B, 250M-500M, 100M-250M, 50M-100M, 25M-50M, 10M-25M
- **Thumbnails**: Visual video cards
- **Badges**: Billionaires Club, Milestone Progress, Highly Viral
- **Drill-Down**: Click video → Historical analysis

### Milestone Tracker
- **Progress Bars**: Visual completion percentage
- **Forecasting**: Days to next milestone (linear projection)
- **Velocity Metrics**: Daily view growth rate
- **5% Threshold**: Only shows videos within striking distance

### Video Deep Dive (Drill-Down)
- **30-Day History**: Plotly time-series chart
- **Growth Metrics**: Views gained, daily growth, growth rate %
- **Engagement Analysis**: Like-to-view, comment-to-view, like-to-comment ratios
- **Metadata**: Duration, category, days since publish

---

## Future Improvements

This project is currently in an early stage, and we've identified several areas that will take the idea to the next level. Our primary focus for future development is organized into the following categories:


**User Experience and Interface (UI/UX)**

Platform Migration: Migrate the frontend from Streamlit to a more robust framework like Next.js. This will provide greater customization and control over the visualization experience, enabling a more professional and scalable user interface.

Custom Watchlists: Implement features for custom channel/video watchlists and easier input methods. This will facilitate more efficient control over the data being tracked and analyzed.

Browser Integration: Explore browser extensions or tools for seamless integration and data input directly from video platforms.


**Additional Data Integration**

To increase the dimensions and insights that can be derived, we plan to integrate data from diverse external sources:

Social Media: Integrate data from platforms like the Twitter API to analyze social engagement surrounding video content.

Contextual Data: Incorporate data from Google Trends and Wikipedia to provide richer context and external factors influencing the trends being analyzed.

Other Platforms: Investigate integrating data from platforms like Spotify for broader media analysis.


**Data Orchestration and Scalability**

While the current setup is functional, migrating to a dedicated orchestration platform will be necessary for enterprise-level scale:

Orchestration Migration: Transition the data pipeline from GitHub Actions to a more comprehensive workflow management platform, such as Apache Airflow. This will allow for greater resilience, more complex and high-level data transformations, and improved monitoring for a scalable, production environment.

---

## Project Documentation

### Key Components

#### 1. **Pipeline Scripts** (`pipeline/`)
- `ingest_youtube_data.py`: Fetches data from YouTube API
- `transform_silver.py`: Cleans and computes metrics
- `aggregate_gold.py`: Creates analytics-ready views
- `setup_database.py`: Initializes tables and schemas

#### 2. **Dashboard** (`dashboard.py`)
- Streamlit multi-page application
- Polars for fast data manipulation
- Plotly for interactive visualizations
- Custom CSS for Social Blade-inspired styling

#### 3. **Database Schema** (`schema.sql`)
```sql
CREATE TABLE channels_log_v3 (
    channel_id VARCHAR(255) NOT NULL,
    channel_title VARCHAR(500),
    description TEXT,
    custom_url VARCHAR(255),
    published_at TIMESTAMP,
    country VARCHAR(10),
    subscriber_count BIGINT,
    view_count BIGINT,
    video_count INTEGER,
    thumbnail_url TEXT,
    ingestion_timestamp TIMESTAMP NOT NULL,
    PRIMARY KEY (channel_id, ingestion_timestamp)
);

CREATE INDEX idx_channels_timestamp ON channels_log_v3(ingestion_timestamp);
```

#### 4. **GitHub Actions** (`.github/workflows/`)
- `pipeline_v3.yml`: 8-hour scheduled runs
- `pipeline_manual.yml`: On-demand testing
- Separate workflows for V1, V2, V3 (isolation)

---

## Testing

### Current Test Coverage
- ✅ API connection validation
- ✅ Database connectivity tests
- ✅ Data transformation logic
- ✅ Metric calculation accuracy
- ✅ Dashboard component rendering

**Run tests:**
```bash
pytest tests/ -v
```

---

## Lessons Learned

### 1. API Quota Management
- **Myth**: Initial estimates suggested 2,719 units/run
- **Reality**: Actual usage is only 77 units/run (35x overestimated)
- **Takeaway**: Batch requests and selective fields dramatically reduce quota consumption

### 2. Data Preservation Strategy
- **Challenge**: Need to iterate pipeline without losing historical data
- **Solution**: Versioned tables (V1 → V2 → V3) preserve all past work
- **Benefit**: Can A/B test transformations against legacy data

### 3. Timestamp Formatting
- **Issue**: ISO 8601 timestamps hard to query and visualize
- **Fix**: Standardize to `YYYY-MM-DD HH:MM:SS` at Silver layer
- **Impact**: 10x faster dashboard queries, cleaner charts

### 4. Database vs. GitHub Secrets
- **Problem**: GitHub Secrets limited to 25-30 channels (64KB limit)
- **Solution**: Move channel list to database `channels_list` table
- **Result**: Now supports 100+ channels with zero code changes

---

<div align="center">
    <strong>Built with ❤️ by the STREAMSMITHS</strong><br>
    <em>IDS 706 Fall 2025 - Duke University</em>
</div>
