# BehavioralEdge — VS Code Setup

Quantitative trading psychology analysis platform.

## Project Structure

```
behavioraledge/
├── pipeline.py          # ML pipeline — run this first
├── app.py               # Streamlit UI — run this after pipeline
├── requirements.txt     # Python dependencies
├── .env                 # Your API keys (never commit this)
└── behavioral_analysis/ # Auto-created by pipeline
    ├── trades.db
    ├── enriched_trades.csv
    ├── behavioral_report.json
    ├── chroma_db/
    └── cache/
```

## Setup

**1. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Groq API key**

Edit `.env`:
```
GROQ_API_KEY=gsk_your_key_here
```
Free key at: https://console.groq.com/keys

## Usage

**Step 1 — Run the pipeline on your trades CSV**
```bash
python pipeline.py --csv your_trades.csv
```

Your CSV needs these columns (names are auto-detected):
- `date` or `timestamp`
- `symbol` or `ticker`
- `side` or `buy_sell` (values: BUY/SELL)
- `quantity` or `qty`
- `price`
- `pnl` (required if CSV only has SELL rows)

**Step 2 — Launch the UI**
```bash
streamlit run app.py
```

Opens at http://localhost:8501

## CSV Formats Supported

**Format A — Full order log (BUY + SELL rows)**
```
timestamp,symbol,side,quantity,price
2024-01-05,RELIANCE.NS,BUY,10,2480
2024-01-10,RELIANCE.NS,SELL,10,2510
```
Pipeline reconstructs PnL via FIFO matching.

**Format B — Closed trades only (SELL rows with PnL)**
```
timestamp,symbol,side,quantity,price,pnl,holding_duration
2024-01-10,RELIANCE.NS,SELL,10,2510,300,5.2
```
Pipeline loads directly, no FIFO needed.

## Re-running

If you update your CSV, re-run the pipeline:
```bash
python pipeline.py --csv updated_trades.csv
```

Use `--skip-market` to skip yfinance fetch and use cached data:
```bash
python pipeline.py --csv trades.csv --skip-market
```
