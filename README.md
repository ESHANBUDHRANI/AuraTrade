⬡ AuraTrade
Quantitative Risk Attribution & Behavioral AI Terminal

AuraTrade is an open-source algorithmic terminal that analyzes your trading psychology. By ingesting raw execution logs, it uses Deep Learning and Explainable AI (XAI) to map your emotional state, isolate behavioral biases, and simulate your digital twin to optimize your edge across shifting market regimes.

✨ Core Modules & Features
🧠 1. Behavioral Digital Twin (HMM Simulation)
AuraTrade doesn't just look at the past; it simulates your future.

Fits a Gaussian Hidden Markov Model (HMM) to your emotional score and position-sizing deviations.

Runs Monte Carlo simulations to project your expected equity curve and maximum drawdown over the next 50 trades, assuming your current behavioral habits persist.

🔬 2. Explainable AI (XAI) Financial Attribution
Understand exactly why a trade bled capital, mapped to specific features.

Trains a Random Forest Regressor on your execution data.

Uses SHAP (SHapley Additive exPlanations) to break down your worst drawdowns, attributing exact percentage drag to factors like "Entered during high volatility regime" or "Position size was 2.5x standard deviation."

🚨 3. Deep Anomaly & Revenge Trade Detection
Detects when you are trading "on tilt" or executing impulsively.

Combines an Isolation Forest with a PyTorch Deep Autoencoder.

Reconstructs every trade vector; trades with high reconstruction errors are flagged as behavioral anomalies (e.g., revenge trading after a severe loss or exiting early due to anxiety).

💬 4. Alpha RAG Agent (LLM Integration)
Chat directly with your quantitative data.

Uses ChromaDB and SentenceTransformers to vectorize your risk profile, VaR limits, and historical performance.

Powered by Llama-3 70B via the Groq API.

Ask counterfactuals: "What would my PnL be if I halved my position size after a loss?" The agent answers strictly using your mathematical context.

📈 5. Automated Market Enrichment & PnL Reconstruction
Zero manual tagging required. Just upload your raw broker logs.

FIFO Reconstruction: Automatically calculates realized PnL, holding durations, and average entry/exit prices from raw BUY/SELL logs.

Market Context: Fetches historical OHLCV data via yfinance and appends 15+ technical indicators (RSI, MACD, ATR, Bollinger Bands).

Regime Classification: Automatically tags the market environment of each trade (e.g., trending_bullish, risk_off, high_volatility).

🎯 6. Dynamic Risk & Bias Profiling
Maps human flaws to mathematical constants.

Prospect Theory Calculation: Calculates your specific Loss Aversion Lambda (λ) and Risk Seeking Alpha (α).

Disposition Effect: Measures your tendency to cut winners short while letting losers run.

Tail Risk Metrics: Calculates Global 95% Value at Risk (VaR) and Conditional VaR (CVaR).

Systematic Recommendations Engine: Auto-generates actionable trading rules based on your specific detected flaws.

⚡ 7. FastAPI Real-Time Webhook
Comes with a built-in uvicorn server to ingest live trades.

Can be connected directly to broker webhooks (Interactive Brokers, Zerodha, Alpaca) to automatically log executions and re-run anomaly detection in real-time.

🛠️ Tech Stack
Frontend: Streamlit, Plotly

Backend: FastAPI, Pandas, SQLite

Machine Learning: PyTorch, Scikit-Learn, SHAP, hmmlearn, pgmpy, arch

LLM & RAG: LangChain, Groq, ChromaDB, HuggingFace SentenceTransformers
