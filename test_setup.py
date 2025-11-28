"""
Setup validation tests for STREAMWATCH V3 Pipeline
Catches configuration errors before pipeline execution
"""

import os
from dotenv import load_dotenv
import requests


def test_env_variables_exist():
    """Check that required environment variables are set"""
    load_dotenv()
    
    required_vars = ["YOUTUBE_API_KEY", "NEON_DATABASE_URL", "YOUTUBE_CHANNEL_IDS"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    assert not missing, f"Missing required environment variables: {missing}"
    print("✓ All required environment variables are set")


def test_youtube_api_key_valid():
    """Verify YouTube API key works with a simple request"""
    load_dotenv()
    
    api_key = os.getenv("YOUTUBE_API_KEY")
    assert api_key, "YOUTUBE_API_KEY not set"
    
    # Simple quota-friendly test: get video categories
    url = "https://www.googleapis.com/youtube/v3/videoCategories"
    params = {"part": "snippet", "regionCode": "US", "key": api_key}
    
    response = requests.get(url, params=params, timeout=10)
    assert response.status_code == 200, f"YouTube API key invalid: {response.status_code}"
    print("✓ YouTube API key is valid")


def test_database_connection():
    """Verify database connection string works"""
    load_dotenv()
    
    db_uri = os.getenv("NEON_DATABASE_URL")
    assert db_uri, "NEON_DATABASE_URL not set"
    
    # Clean connection string
    db_uri = db_uri.strip().strip("'\"")
    if db_uri.startswith("psql "):
        db_uri = db_uri[5:].strip()
    
    # Test connection with SQLAlchemy
    from sqlalchemy import create_engine, text
    
    engine = create_engine(db_uri)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.fetchone()[0] == 1
    
    print("✓ Database connection successful")


def test_channel_ids_format():
    """Verify YOUTUBE_CHANNEL_IDS is properly formatted"""
    load_dotenv()
    
    channel_ids_str = os.getenv("YOUTUBE_CHANNEL_IDS")
    assert channel_ids_str, "YOUTUBE_CHANNEL_IDS not set"
    
    channel_ids = [c.strip() for c in channel_ids_str.split(",") if c.strip()]
    assert len(channel_ids) > 0, "YOUTUBE_CHANNEL_IDS is empty"
    
    # Check format (YouTube channel IDs are 24 characters)
    for cid in channel_ids:
        assert len(cid) == 24, f"Invalid channel ID format: {cid}"
    
    print(f"✓ Found {len(channel_ids)} valid channel IDs")


def test_output_directories():
    """Verify output directories can be created"""
    from pathlib import Path
    
    dirs = ["bronze_data_v3", "silver_data_v3"]
    for dir_name in dirs:
        path = Path(dir_name)
        path.mkdir(parents=True, exist_ok=True)
        assert path.exists() and path.is_dir()
    
    print("✓ Output directories accessible")


if __name__ == "__main__":
    print("=" * 60)
    print("STREAMWATCH V3 - Setup Validation Tests")
    print("=" * 60)
    
    tests = [
        test_env_variables_exist,
        test_youtube_api_key_valid,
        test_database_connection,
        test_channel_ids_format,
        test_output_directories,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n[TEST] {test.__doc__}")
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        exit(1)