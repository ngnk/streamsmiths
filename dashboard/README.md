# Dashboard

## How to run

1. CD into /dashboard directory and install dependencies:
```bash
pip install -r requirements.txt
```

2. Create .env file with your database connection string with NEON_DATABASE_URL as the variable name. If you use a separate service, feel free to change the variable name in dashboard script.

3. Run:
```bash
streamlit run dashboard.py
```
