<div align="center">

# STREAMWATCH

### IDS 706 Fall 2025 Final Project: YouTube Analytics Platform

![V3](https://github.com/ngnk/streamsmiths/actions/workflows/workflow_v3.yml/badge.svg) ![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub%20Actions-brightgreen.svg)

**Team 6 (Streamsmiths): Tony Ngari, Can He, Matthew Fischer, Joseph Hong, Trey Chase**

</div>

## Table of Contents

- [Project Overview](#-Project-Overview)
- [Architecture](#-Architecture)
- [Setup Instructions](#-Setup)
- [Dashboard](#-Dashboard)
- [Key Principles](#-Key-Principles)
- [Future Improvements](#-future-improvements)

---

# Project Overview

**STREAMWATCH** is a comprehensive YouTube analytics platform that provides insights into channel performance, video trends, and milestone tracking.

The platform processes data from 50+ YouTube channels through a **bronze-silver-gold data pipeline architecture**, delivering:
- Real-time channel and video performance tracking
- Milestone achievement monitoring
- Engagement analytics and trend identification
- Historical data analysis with interactive visualizations
- Scalable data infrastructure supporting future ML predictions

### Team

| Name | Role | Responsibilities |
|------|------|-----------------|
| **Tony N.** | Leader / Engineering | Pipeline architecture, database design, system integration, project coordination |
| **Joseph H.** | Engineering | Data processing workflows, API integration, model development |
| **Trey C.** | Data Science | Model development, metrics calculation |
| **Can H.** | Analytics | Dashboard development, visualization design |
| **Matthew F.** | Data Science | Model development, metrics calculation |

---

# Architecture

STREAMWATCH implements a **medallion architecture** (bronze-silver-gold).

### Data Source
[**YouTube Data API v3**](https://developers.google.com/youtube/v3)

**Bronze Layer (bronze_v3.py)**

- Grabs raw data from YouTube (channel info, videos, trending content) and saves as JSON files with timestamps

**Silver Layer (silver_v3.py)**

- Cleans up the raw data, Calculates useful metrics like engagement rates, milestons, then converts JSON to Parquet (a faster file format)

**Gold Layer (gold_v3.py)**

- Takes cleaned data and loads it into the database

**Orchestration (run_pipeline_v3.py)**

- Runs all three layers in order automatically
- Can be scheduled to run hourly via GitHub Actions

**Visualization**

- Interactive web dashboard displaying channnel and video metrircs.

---

# Setup

**Get a YouTube API Key:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable "YouTube Data API v3"
4. Create credentials (API key)
5. Copy the API key to `.env`

**Set up Neon Database:**
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

### 4. Confirm all componenets are good-to-go
```bash
python test_setup.py
```

### 5. Initialize Database Tables
Run the pipeline setup script to create tables:

```bash
python scripts/setup_database.py
```

This will create:
- `channels_log_v3` (Bronze layer)
- `videos_log_v3` (Bronze layer)
- Future: Silver and Gold layer tables

### 6. Run Initial Data Ingestion
```bash
python pipeline/ingest_youtube_data.py
```

### 7. Launch Dashboard
```bash
cd dashboard
streamlit run dashboard.py
```

Access the dashboard at `http://localhost:8501`

### 8. (Optional) Set Up GitHub Actions
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

# Dashboard

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

# Key Principles

### 1. Scalability
- **Horizontal Scaling**: Database-driven channel management supports 50-100+ channels (vs. 25-channel GitHub Secrets limit)
- **API Quota Efficiency**: Pipeline consumes only 77 units/run (35x under estimated 2,719), providing massive headroom
- **Time-Series Architecture**: Append-only Bronze tables support unlimited historical growth
- **Cloud Database**: Neon PostgreSQL with connection pooling handles concurrent queries

### 2. Modularity
- **Versioned Tables**: Separate V1, V2, V3 schemas preserve existing data during iteration
- **Layered Architecture**: Bronze (raw) → Silver (transformed) → Gold (analytics) separation
- **Reusable Functions**: `calculate_grade()`, `format_number()`, `load_channels()` used across dashboard
- **Independent Workflows**: Separate GitHub Actions for V1, V2, V3 pipelines

### 3. Reusability
- **Templated SQL Queries**: Parameterized queries work for channels, videos, time ranges
- **Abstracted Data Loaders**: `load_video_history()`, `load_channel_history()` functions
- **Style Components**: Reusable CSS classes (`.metric-card`, `.channel-card`, `.milestone-badge`)
- **Visualization Templates**: Plotly chart configurations used across multiple pages

### 4. Observability
- **Ingestion Timestamps**: Every record tagged with `ingestion_timestamp` for lineage tracking
- **Pipeline Logging**: GitHub Actions logs capture API responses, row counts, errors
- **Version Tracking**: V1 → V2 → V3 tables preserve evolution history
- **Dashboard Metrics**: Real-time counts of channels, videos, Billionaires Club members

### 5. Data Governance
- **Schema Versioning**: V1, V2, V3 tables document pipeline evolution
- **Immutable Bronze Layer**: Raw API responses never modified (append-only)
- **Data Lineage**: Clear transformation path: Bronze → Silver → Gold
- **Quality Validation**: Timestamp formatting, duplicate detection, null handling

### 6. Reliability
- **Connection Pooling**: SQLAlchemy `pool_pre_ping=True` prevents stale connections
- **Error Recovery**: Try-catch blocks in API calls with graceful degradation
- **Scheduled Automation**: GitHub Actions hourly cron ensures consistent data freshness
- **Caching Strategy**: Streamlit `@st.cache_data(ttl=3600)` reduces database load


### 7. Efficiency
- **API Quota Optimization**: Batch requests, selective field retrieval (`part='snippet,statistics'`)
- **Query Optimization**: `DISTINCT ON` for latest records, indexed `ingestion_timestamp`
- **Dashboard Caching**: 1-hour TTL prevents redundant database queries
- **Selective Data Loading**: Only fetch 20 videos per page, 30-day history windows

### 8. Security
- **Secret Management**: API keys stored in GitHub Secrets, never committed to Git
- **Environment Variables**: `.env` file in `.gitignore`, `python-dotenv` for local dev
- **Database Encryption**: Neon provides SSL/TLS connections by default
- **No Hardcoded Credentials**: All sensitive data externalized to environment config

---

## Future Improvements

This project is currently in an early stage, and we've identified several areas that will take the idea to the next level. Our primary focus for future development is organized into the following categories:

**User Experience and Interface (UI/UX)**
- Migrate the frontend from Streamlit to a more robust framework like Next.js. This will provide greater customization and control over the visualization experience, enabling a more professional and scalable user interface.
- Implement features for custom channel/video watchlists and easier input methods. This will facilitate more efficient control over the data being tracked and analyzed.
- Explore browser extensions or tools for seamless integration and data input directly from video platforms.

**Additional Data Integration**
- Integrate data from platforms like the Twitter API and Spotify to analyze social engagement surrounding video content.
- Incorporate data from Google Trends and Wikipedia to provide richer context and external factors influencing the trends being analyzed.


**Data Orchestration and Scalability**
- Transition the data pipeline from GitHub Actions to a more comprehensive workflow management platform, such as Apache Airflow. This will allow for greater resilience, more complex and high-level data transformations, and improved monitoring for a scalable, production environment.

---

<div align="center">
    <strong>Built with ❤️ by the STREAMSMITHS</strong><br>
    <em>IDS 706 Fall 2025 - Duke University</em>
</div>
