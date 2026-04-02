"""
BehavioralEdge — Interactive Streamlit UI
Run: streamlit run app.py
"""

import os, json, tempfile, subprocess
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

BASE_PATH = Path("behavioral_analysis")

st.set_page_config(
    page_title="BehavioralEdge Terminal",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp { background-color: #080c10 !important; color: #c9d4e0 !important; font-family: 'DM Mono', monospace !important; }
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
.block-container { padding: 2rem 2.5rem !important; max-width: 100% !important; }

.be-header { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #1a2332; padding-bottom: 1.2rem; margin-bottom: 1.5rem; }
.be-logo { font-family: 'Syne', sans-serif; font-size: 1.25rem; font-weight: 800; letter-spacing: 0.08em; color: #e8f0fe; display: flex; align-items: center; gap: 10px; }
.be-logo span { color: #3b82f6; }
.be-tagline { font-size: 0.7rem; color: #4a6580; letter-spacing: 0.15em; text-transform: uppercase; margin-top: 4px; }
.be-status { display: flex; align-items: center; gap: 12px; }
.be-pill { font-size: 0.62rem; letter-spacing: 0.12em; text-transform: uppercase; padding: 4px 12px; border-radius: 20px; font-weight: 500; }
.be-pill-green { background: rgba(34,197,94,0.1); color: #22c55e; border: 1px solid rgba(34,197,94,0.2); }
.be-pill-red { background: rgba(239,68,68,0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.2); }

.kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 2rem; }
.kpi-card { background: #111a26; padding: 1.2rem 1.5rem; border-radius: 10px; border: 1px solid #1a2d42; transition: transform 0.2s, border-color 0.2s; }
.kpi-card:hover { border-color: #3b82f6; transform: translateY(-2px); }
.kpi-label { font-size: 0.65rem; letter-spacing: 0.15em; text-transform: uppercase; color: #6b8db0; margin-bottom: 0.5rem; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; color: #e8f0fe; line-height: 1.2; }
.kpi-sub { font-size: 0.7rem; margin-top: 0.3rem; }
.kpi-pos { color: #22c55e; } .kpi-neg { color: #ef4444; } .kpi-neu { color: #3b82f6; }

div[data-testid="stSidebar"] { background-color: #0b121b !important; border-right: 1px solid #1a2d42 !important; }
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #1a2332 !important; }
.stTabs [data-baseweb="tab"] { color: #4a6580 !important; font-family: 'DM Mono', monospace !important; font-size: 0.7rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; border: none !important; padding: 0.8rem 1.2rem !important; }
.stTabs [aria-selected="true"] { color: #3b82f6 !important; border-bottom: 2px solid #3b82f6 !important; }

/* ── CHATBOT ── */
.chat-wrap {
    background: #0d1520;
    border: 1px solid #1a2d42;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.chat-messages {
    overflow-y: auto;
    padding: 0.5rem 0;
    max-height: 400px;
    min-height: 120px;
    scrollbar-width: thin;
    scrollbar-color: #1a2332 transparent;
}
.msg { display: flex; gap: 10px; margin-bottom: 1rem; align-items: flex-start; }
.msg-user { flex-direction: row-reverse; }
.msg-avatar {
    width: 30px; height: 30px; border-radius: 6px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 600;
}
.avatar-ai { background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.2); }
.avatar-user { background: rgba(139,92,246,0.15); color: #8b5cf6; border: 1px solid rgba(139,92,246,0.2); }
.msg-bubble { font-size: 0.8rem; line-height: 1.6; padding: 0.8rem 1.2rem; max-width: 85%; }
.bubble-ai { background: #111a26; border: 1px solid #1a2d42; color: #c9d4e0; border-radius: 2px 12px 12px 12px; }
.bubble-user { background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.15); color: #c9d4e0; border-radius: 12px 2px 12px 12px; }

.panel-title { font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase; color: #4a6580; margin-bottom: 1rem; padding-bottom: 0.8rem; border-bottom: 1px solid #111a26; }

/* ── BUTTONS ── */
.stButton > button {
    background: #0d1520 !important; color: #6b8db0 !important;
    border: 1px solid #1a2d42 !important; border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.7rem !important;
    padding: 0.5rem 0.8rem !important; transition: all 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover { background: #111e2e !important; color: #3b82f6 !important; border-color: rgba(59,130,246,0.4) !important; }

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] textarea {
    background: #0d1520 !important; border: 1px solid #1a2d42 !important;
    color: #c9d4e0 !important; border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        with open(BASE_PATH / 'behavioral_report.json') as f:
            report = json.load(f)
        df = pd.read_csv(BASE_PATH / 'enriched_trades.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return report, df
    except:
        return None, None

@st.cache_resource
def init_chroma():
    try:
        client = chromadb.PersistentClient(path=str(BASE_PATH / "chroma_db"))
        emb = SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
        return (
            client.get_collection('session_data',     embedding_function=emb),
            client.get_collection('static_knowledge', embedding_function=emb),
            True
        )
    except:
        return None, None, False

# ─────────────────────────────────────────
# UPLOAD GATE
# ─────────────────────────────────────────
if 'data_loaded' not in st.session_state:
    report, df = load_data()
    st.session_state.data_loaded = bool(report and df is not None)
    if st.session_state.data_loaded:
        st.session_state.report = report
        st.session_state.df     = df

if not st.session_state.data_loaded:
    st.markdown("""
    <div style='display:flex;align-items:center;justify-content:center;margin-top:5rem;flex-direction:column;gap:1.5rem'>
        <div style='font-family:Syne,sans-serif;font-size:3.5rem;color:#1a2d42'>⬡</div>
        <div style='font-size:0.8rem;color:#6b8db0;letter-spacing:0.2em;text-transform:uppercase'>Upload Trades to Initialize Terminal</div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("", type=["csv"])
        if uploaded_file is not None:
            with st.spinner("Executing Quantitative Pipeline..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    result = subprocess.run(
                        ["python", "pipeline.py", "--csv", tmp_path],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        report, df = load_data()
                        st.session_state.report      = report
                        st.session_state.df          = df
                        st.session_state.data_loaded = True
                        st.rerun()
                    else:
                        st.error("Pipeline failed to execute.")
                        with st.expander("Show Error Log"):
                            st.code(result.stderr)
                finally:
                    os.remove(tmp_path)
    st.stop()

report        = st.session_state.report
df            = st.session_state.df
b             = report['biases']
session_coll, static_coll, db_ok = init_chroma()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.markdown('<div class="be-logo" style="margin-bottom:2rem;font-size:1rem;">⬡ <span>Filters</span></div>', unsafe_allow_html=True)
selected_symbol = st.sidebar.selectbox("Ticker Symbol", ['ALL'] + sorted(df['symbol'].unique().tolist()))
selected_side   = st.sidebar.selectbox("Trade Direction", ['ALL'] + sorted(df['side'].unique().tolist()))
min_date, max_date = df['timestamp'].min().date(), df['timestamp'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
groq_key   = os.getenv("GROQ_API_KEY", "") or st.sidebar.text_input("Groq API Key", type="password", placeholder="gsk_...")
llm        = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.0, api_key=groq_key) if groq_key else None

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset & Upload New CSV", use_container_width=True):
    st.session_state.clear()
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# ─────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────
filtered_df = df.copy()
if selected_symbol != 'ALL':
    filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]
if selected_side != 'ALL':
    filtered_df = filtered_df[filtered_df['side'] == selected_side]
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['timestamp'].dt.date >= date_range[0]) &
        (filtered_df['timestamp'].dt.date <= date_range[1])
    ]

# ─────────────────────────────────────────
# PLOTLY BASE THEME — no margin here, apply per chart
# ─────────────────────────────────────────
PLOT_BASE = dict(
    template='plotly_dark',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Mono', color='#6b8db0', size=11),
    legend=dict(bgcolor='rgba(0,0,0,0)', orientation="h",
                yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=40, b=20)   # ← single definition here, never repeated
)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown(f"""
<div class="be-header">
    <div>
        <div class="be-logo">⬡ <span>Behavioral</span>Edge</div>
        <div class="be-tagline">Interactive Alpha Terminal</div>
    </div>
    <div class="be-status">
        <div class="be-pill {'be-pill-green' if db_ok else 'be-pill-red'}">{'VECTOR STORE OK' if db_ok else 'VECTOR STORE OFFLINE'}</div>
        <div class="be-pill {'be-pill-green' if groq_key else 'be-pill-red'}">{'AI ONLINE' if groq_key else 'AI OFFLINE'}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────
dyn_pnl    = filtered_df['pnl'].sum()
dyn_trades = len(filtered_df)
dyn_wr     = (filtered_df['pnl'] > 0).mean() if dyn_trades > 0 else 0
winners    = filtered_df[filtered_df['pnl'] > 0]['pnl']
losers     = filtered_df[filtered_df['pnl'] < 0]['pnl']
rr_ratio   = abs(winners.mean() / losers.mean()) if len(losers) > 0 and losers.mean() != 0 else 0

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Filtered PnL</div>
        <div class="kpi-value {'kpi-pos' if dyn_pnl >= 0 else 'kpi-neg'}">₹{dyn_pnl:,.0f}</div>
        <div class="kpi-sub">Across {dyn_trades} trades</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Win Rate</div>
        <div class="kpi-value">{dyn_wr*100:.1f}%</div>
        <div class="kpi-sub {'kpi-pos' if dyn_wr > 0.5 else 'kpi-neg'}">{'Above' if dyn_wr > 0.5 else 'Below'} breakeven</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Risk / Reward</div>
        <div class="kpi-value">{rr_ratio:.2f}x</div>
        <div class="kpi-sub">Avg Win / Avg Loss</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Global 95% VaR</div>
        <div class="kpi-value">{report['risk_profile']['var95']:.3f}</div>
        <div class="kpi-sub kpi-neg">Tail Risk</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Behavioral Style</div>
        <div class="kpi-value" style="font-size:1.05rem;padding-top:0.4rem">{report['behavioral_profile']['trading_style'].replace('_', ' ').title()}</div>
        <div class="kpi-sub kpi-neu">{report['behavioral_profile']['dominant_cluster'].replace('_', ' ').title()}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────
col_charts, col_ai = st.columns([1.8, 1], gap="large")

with col_charts:
    if dyn_trades == 0:
        st.warning("No trades found for this filter combination.")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Performance", "Forensics", "Regimes", "Data Explorer", "XAI & Digital Twin"
        ])

        with tab1:
            c1, c2 = st.columns(2)
            filtered_df = filtered_df.sort_values('timestamp').copy()
            filtered_df['cumulative_pnl'] = filtered_df['pnl'].cumsum()

            fig1 = px.line(filtered_df, x='timestamp', y='cumulative_pnl',
                           title="Cumulative Equity Curve")
            fig1.update_traces(line_color='#3b82f6', fill='tozeroy',
                               fillcolor='rgba(59,130,246,0.08)')
            fig1.add_hline(y=0, line_dash='dot', line_color='#4a6580')
            fig1.update_layout(**PLOT_BASE, xaxis_title="", yaxis_title="₹ PnL")
            c1.plotly_chart(fig1, use_container_width=True)

            fig2 = px.bar(filtered_df, x='timestamp', y='pnl',
                          color='emotional_state', title="PnL by Emotional State",
                          color_discrete_map={'calm':'#22c55e','anxious':'#f59e0b','euphoric':'#ef4444'})
            fig2.update_layout(**PLOT_BASE, xaxis_title="")
            c2.plotly_chart(fig2, use_container_width=True)

        with tab2:
            c3, c4 = st.columns(2)
            fig3 = px.scatter(
                filtered_df, x='holding_duration', y='pnl',
                color='is_loss',
                size=filtered_df['position_value'].clip(lower=1),
                hover_data=['symbol'],
                title="Holding Duration vs Result",
                color_continuous_scale=[[0, '#22c55e'], [1, '#ef4444']]
            )
            fig3.update_layout(**PLOT_BASE, coloraxis_showscale=False)
            fig3.add_hline(y=0, line_dash='dot', line_color='#4a6580')
            c3.plotly_chart(fig3, use_container_width=True)

            r_vals = [
                min(b['loss_aversion_lambda'] / 3.0, 1.0),
                min(b['disposition_score'] / 2.0, 1.0),
                min(b['revenge_trading_rate'] * 2.0, 1.0),
                min(b['early_exit_rate'] * 2.0, 1.0),
                min(report['anomaly']['anomaly_rate'] * 5.0, 1.0)
            ]
            r_cats = ['Loss Aversion','Disposition','Revenge','Early Exit','Anomalies']
            fig4 = go.Figure(go.Scatterpolar(
                r=r_vals + [r_vals[0]], theta=r_cats + [r_cats[0]],
                fill='toself', fillcolor='rgba(139,92,246,0.15)',
                line=dict(color='#8b5cf6', width=2),
                marker=dict(size=6, color='#8b5cf6')
            ))
            fig4.update_layout(
                **PLOT_BASE, title="Global Bias Radar",
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0,1], gridcolor='#1a2332')
                )
            )
            c4.plotly_chart(fig4, use_container_width=True)

        with tab3:
            c5, c6 = st.columns(2)
            if 'market_regime' in filtered_df.columns:
                fig5 = px.box(filtered_df, x='market_regime', y='pnl',
                              color='market_regime', title="PnL by Market Regime")
                fig5.update_layout(**PLOT_BASE, xaxis_title="", showlegend=False)
                fig5.add_hline(y=0, line_dash='dot', line_color='#4a6580')
                c5.plotly_chart(fig5, use_container_width=True)

            if 'anomaly_score' in filtered_df.columns:
                fig6 = px.scatter(
                    filtered_df, x='timestamp', y='anomaly_score',
                    color='anomaly_flag',
                    size='position_value',
                    hover_data=['symbol', 'pnl'],
                    title="Anomaly Timeline",
                    color_continuous_scale=[[0, '#3b82f6'], [1, '#f59e0b']]
                )
                fig6.update_layout(**PLOT_BASE, coloraxis_showscale=False)
                c6.plotly_chart(fig6, use_container_width=True)

        with tab4:
            display_cols = ['timestamp','symbol','side','price','quantity',
                            'pnl','holding_duration','emotional_state','market_regime']
            st.dataframe(
                filtered_df[[c for c in display_cols if c in filtered_df.columns]],
                use_container_width=True, height=400
            )

        with tab5:
            c7, c8 = st.columns(2)
            with c7:
                st.markdown("<h5 style='color:#e8f0fe;font-family:Syne,sans-serif'>XAI Financial Attribution</h5>", unsafe_allow_html=True)
                xai_data = report.get('xai_attribution', [])
                if xai_data:
                    for t in xai_data[:3]:
                        st.markdown(f"""
                        <div style='background:#111a26;padding:10px;border-radius:8px;border-left:3px solid #ef4444;margin-bottom:8px;'>
                            <div style='font-size:0.8rem;color:#e8f0fe'>Trade {t['trade_index']} | PnL: <span style='color:#ef4444'>₹{t['pnl']:.2f}</span></div>
                            <div style='font-size:0.7rem;color:#6b8db0'>Drivers: {', '.join(list(t['attribution'].keys()))}</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("XAI attribution data not yet generated.")

            with c8:
                st.markdown("<h5 style='color:#e8f0fe;font-family:Syne,sans-serif'>Behavioral Digital Twin</h5>", unsafe_allow_html=True)
                dt_data = report.get('digital_twin')
                if dt_data and 'simulated_equity_curve' in dt_data:
                    fig_dt = px.line(y=dt_data['simulated_equity_curve'], title="HMM Monte Carlo Projection")
                    fig_dt.update_traces(line_color='#8b5cf6', line_dash='dot')
                    # ✅ FIXED: no margin= here, it's already in PLOT_BASE
                    fig_dt.update_layout(**PLOT_BASE, height=200,
                                         xaxis_title="Trade Step", yaxis_title="Proj. PnL")
                    st.plotly_chart(fig_dt, use_container_width=True)
                else:
                    st.info("Digital Twin data not yet generated.")

# ─────────────────────────────────────────
# RAG CHATBOT COLUMN
# ─────────────────────────────────────────
with col_ai:
    st.markdown('<div class="panel-title">⬡ Alpha RAG Agent</div>', unsafe_allow_html=True)

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                f"Analyzed **{report['summary']['total_trades']}** trades. "
                f"Your style is **{report['behavioral_profile']['trading_style'].replace('_', ' ')}**. "
                f"Win rate: **{report['summary']['win_rate']*100:.1f}%**. "
                f"How can I help optimize your edge?"
            )
        }]

    # Quick prompt buttons
    qc1, qc2 = st.columns(2)
    qc3, qc4 = st.columns(2)
    q1 = qc1.button("🚨 Worst biases",     use_container_width=True)
    q2 = qc2.button("📊 Risk profile",     use_container_width=True)
    q3 = qc3.button("💡 Halve my size?",   use_container_width=True)
    q4 = qc4.button("📈 Best regime?",     use_container_width=True)

    # Chat bubble renderer
    chat_html = ""
    for m in st.session_state.messages[-12:]:
        role_cls   = "msg-user" if m["role"] == "user" else ""
        avt_cls    = "avatar-user" if m["role"] == "user" else "avatar-ai"
        avt_lbl    = "U" if m["role"] == "user" else "⬡"
        bubble_cls = "bubble-user" if m["role"] == "user" else "bubble-ai"
        content    = m["content"].replace("<","&lt;").replace(">","&gt;")
        chat_html += f"""
        <div class="msg {role_cls}">
            <div class="msg-avatar {avt_cls}">{avt_lbl}</div>
            <div class="msg-bubble {bubble_cls}">{content}</div>
        </div>"""

    # ✅ FIXED: chat container now has visible background + border
    st.markdown(f'<div class="chat-wrap"><div class="chat-messages">{chat_html}</div></div>',
                unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Ask about biases, risk, or counterfactuals…")
    if q1: prompt = "What are my worst trading biases?"
    if q2: prompt = "Explain my VaR and CVaR risk profile."
    if q3: prompt = "What if I had halved my position size?"
    if q4: prompt = "Which market regime should I focus on?"

    if prompt:
        if not llm:
            st.warning("👈 Add GROQ_API_KEY to .env or enter it in the sidebar.")
        elif not db_ok:
            st.warning("ChromaDB offline — upload a CSV first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Querying vector store..."):
                def retrieve(query, ns=3, nk=2):
                    try:
                        s = session_coll.query(query_texts=[query], n_results=ns)['documents'][0]
                        k = static_coll.query(query_texts=[query],  n_results=nk)['documents'][0]
                        return s + k
                    except:
                        return []

                docs    = retrieve(prompt)
                rag_ctx = '\n'.join([f'- {d}' for d in docs])

                sys_prompt = (
                    f"You are a strict quantitative trading analyst.\n"
                    f"TRADER: style={report['behavioral_profile']['trading_style']}, "
                    f"win_rate={report['summary']['win_rate']:.1%}, "
                    f"loss_aversion_lambda={b['loss_aversion_lambda']:.2f}, "
                    f"disposition={b['disposition_score']:.2f}, "
                    f"best_regime={report['risk_profile']['best_regime']}.\n"
                    f"RULES: Answer ONLY using provided context. "
                    f"Be analytical, max 4 sentences. Bold key numbers."
                )
                try:
                    resp = llm.invoke([
                        SystemMessage(content=sys_prompt),
                        HumanMessage(content=f"Context:\n{rag_ctx}\n\nQuery: {prompt}")
                    ]).content
                    st.session_state.messages.append({"role": "assistant", "content": resp})
                    st.rerun()
                except Exception as e:
                    st.error(f"API Error: {e}")

# ─────────────────────────────────────────
# RECOMMENDATIONS ENGINE
# ─────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div class="panel-title" style="font-size:1rem;">Systematic Recommendations & Next Actions</div>',
            unsafe_allow_html=True)

rec_b     = report['biases']
rec_r     = report['risk_profile']
rec_pnl   = report['summary']['total_pnl']
rec_style = report['behavioral_profile']['trading_style']
suggestions = []

if rec_b['loss_aversion_lambda'] > 2.25:
    suggestions.append({
        "type": "critical", "icon": "⚠️",
        "title": "Hard Stop-Loss Implementation Required",
        "desc": f"Your loss aversion lambda is <b>{rec_b['loss_aversion_lambda']:.2f}</b>. You hold onto losing positions hoping they recover. Switch to automated broker-side bracket orders on your next 10 trades."
    })
elif rec_b['disposition_score'] > 1.5:
    suggestions.append({
        "type": "warning", "icon": "✂️",
        "title": "Let Winners Run",
        "desc": f"Your disposition score is <b>{rec_b['disposition_score']:.2f}</b>. You secure small profits too early. Sell only half at your first target and trail the stop for the remainder."
    })

if rec_b['revenge_trading_rate'] > 0.3:
    suggestions.append({
        "type": "critical", "icon": "🛑",
        "title": "Implement a 24h Cool-Down Rule",
        "desc": f"You execute impulsive trades <b>{rec_b['revenge_trading_rate']*100:.0f}%</b> of the time after a loss. Enforce a strict 24-hour no-trade window after any loss exceeding your daily VaR limit."
    })

if rec_r['worst_regime'] != 'N/A':
    suggestions.append({
        "type": "info", "icon": "📉",
        "title": "Regime Filter Adjustment",
        "desc": f"Your equity bleeds heavily during <b>{rec_r['worst_regime'].replace('_', ' ')}</b> conditions. Reduce your base position size by <b>50%</b> when the market enters this regime."
    })

if rec_style == 'disciplined_systematic' and rec_pnl > 0:
    suggestions.append({
        "type": "success", "icon": "📈",
        "title": "Scale Up Capital Allocation",
        "desc": "Your behavioral cluster shows high systemic discipline with positive expectancy. Consider increasing your overall portfolio risk allocation by <b>0.5%</b> per trade."
    })

if not suggestions:
    st.info("No critical systemic adjustments needed. Continue executing your edge.")
else:
    cols = st.columns(len(suggestions))
    color_map = {
        "critical": ("#1a0b0f", "#ef4444"),
        "warning":  ("#1a130b", "#f59e0b"),
        "info":     ("#0b151a", "#3b82f6"),
        "success":  ("#0b1a10", "#22c55e"),
    }
    for i, rec in enumerate(suggestions):
        bg, border = color_map.get(rec["type"], ("#111a26", "#3b82f6"))
        cols[i].markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-radius:10px;padding:1.5rem;height:100%;">
            <h3 style="margin-top:0;font-family:'Syne',sans-serif;font-size:1rem;color:#e8f0fe;margin-bottom:0.8rem">
                {rec['icon']} {rec['title']}
            </h3>
            <p style="font-size:0.78rem;color:#c9d4e0;line-height:1.7;margin:0">{rec['desc']}</p>
        </div>
        """, unsafe_allow_html=True)