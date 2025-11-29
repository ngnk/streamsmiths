# Dashboard

## How to run

1. CD into /dashboard directory and install dependencies:
```bash
pip install -r requirements.txt
```

2. Create .env file with your database connection string (in our case it is NEON_DATABASE_URL). If you use a separate service change variable name appropriately in dashboard script.

3. Run:
```bash
streamlit run dashboard.py
```
