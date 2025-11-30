#!/usr/bin/env python3
"""
Quick connection test for Neon database
Run this to verify your .env is configured correctly
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

def test_connection():
    print("🔍 Testing Neon Database Connection")
    print("=" * 60)
    
    # Load .env
    load_dotenv()
    
    db_uri = os.getenv("NEON_DATABASE_URL")
    
    if not db_uri:
        print("❌ ERROR: NEON_DATABASE_URL not found in .env file")
        print("\nCreate a .env file with:")
        print("NEON_DATABASE_URL=postgresql://...")
        return False
    
    # Check for channel_binding (common issue)
    if "channel_binding" in db_uri:
        print("⚠️  WARNING: Found 'channel_binding' in URL")
        print("   Pooler connections don't support channel_binding!")
        print("   Remove '&channel_binding=require' from your .env file")
        return False
    
    print(f"📡 Connecting to: {db_uri[:60]}...")
    print()
    
    try:
        # Create engine
        engine = create_engine(db_uri, pool_pre_ping=True)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            
            print("✅ CONNECTION SUCCESSFUL!")
            print()
            print(f"📊 PostgreSQL Version: {version[:50]}...")
            print()
            
            # Test if tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE '%log%'
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result.fetchall()]
            
            if tables:
                print("📋 Available tables:")
                for table in tables:
                    print(f"   • {table}")
            else:
                print("⚠️  No tables found (this might be okay if database is new)")
            
            print()
            print("🎉 Database connection is working perfectly!")
            return True
            
    except Exception as e:
        print("❌ CONNECTION FAILED!")
        print()
        print(f"Error: {e}")
        print()
        print("💡 Common solutions:")
        print("   1. Remove '&channel_binding=require' from .env")
        print("   2. Check username/password are correct")
        print("   3. Verify you have network access to Neon")
        print("   4. Make sure .env file is in the same directory")
        return False

if __name__ == "__main__":
    success = test_connection()
    exit(0 if success else 1)