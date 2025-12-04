# Dashboard

## How to run

1. CD into /dashboard directory and install dependencies:
```bash
cd dashboard
pip install -r requirements.txt
```

2. Create .env file with your database connection string with NEON_DATABASE_URL as the variable name
```bash
NEON_DATABASE_URL='YOUR_CONNECTION_STRING'
```

3. Run:
```bash
streamlit run dashboard_v3.py
```
The dashboard should automatically open in your browser at `http://localhost:8501`

4. Operate:
Do not close the terminal window otherwise dashboard will terminate.
To terminate, press CTRL + C in terminal.
