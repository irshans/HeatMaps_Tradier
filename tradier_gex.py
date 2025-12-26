import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

# --- APP CONFIG ---
st.set_page_config(page_title="GEX Pro 2025", page_icon="📊", layout="wide")

# --- CREDENTIALS ---
if "TRADIER_TOKEN" in st.secrets:
    TRADIER_TOKEN = st.secrets["TRADIER_TOKEN"]
else:
    st.error("Please set TRADIER_TOKEN in Streamlit Secrets.")
    st.stop()

BASE_URL = "https://api.tradier.com/v1/"
CONTRACT_SIZE = 100

st.markdown("""
    <style>
    .block-container { padding-top: 24px; padding-bottom: 8px; }
    button[kind="primary"], .stButton>button { padding:4px 8px !important; font-size:12px !important; height:30px !important; }
    input[type="text"], input[type="number"] { padding:6px 8px !important; font-size:12px !important; height:28px !important; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# Tradier API Functions
# -------------------------
def tradier_get(endpoint, params):
    headers = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers)
        if response.status_code == 200: return response.json()
    except: pass
    return None

@st.cache_data(ttl=3600)
def get_open_market_days():
    """Builds a whitelist from the official Tradier Calendar."""
    open_days = set()
    cal_data = tradier_get("markets/calendar", {})
    try:
        if cal_data and 'calendar' in cal_data:
            days = cal_data['calendar']['days']['day']
            if isinstance(days, dict): days = [days]
            for d in days:
                if d.get('status') == 'open':
                    open_days.add(d.get('date'))
    except: pass
    return open_days

def fetch_tradier_data(ticker, max_exp):
    open_days = get_open_market_days()
    
    # 1. Get Spot Price
    quote_data = tradier_get("markets/quotes", {"symbols": ticker})
    if not quote_data or 'quotes' not in quote_data: return None, None
    quote = quote_data['quotes']['quote']
    S = float(quote['last']) if isinstance(quote, dict) else float(quote[0]['last'])

    # 2. Get Expirations
    exp_data = tradier_get("markets/options/expirations", {"symbol": ticker, "includeAllRoots": "true"})
    if not exp_data or 'expirations' not in exp_data or not exp_data['expirations']: 
        return S, None
    
    all_exps = exp_data['expirations']['date']
    if not isinstance(all_exps, list): all_exps = [all_exps]
    
    # 3. STRICT CALENDAR FILTER
    valid_exps = [d for d in all_exps if d in open_days]
    target_exps = valid_exps[:max_exp]
    
    dfs = []
    prog = st.progress(0, text="Fetching Live Option Chains...")
    for i, exp in enumerate(target_exps):
        chain = tradier_get("markets/options/chains", {"symbol": ticker, "expiration": exp, "greeks": "true"})
        if chain and 'options' in chain and chain['options']:
            opts = chain['options']['option']
            dfs.append(pd.DataFrame(opts) if isinstance(opts, list) else pd.DataFrame([opts]))
        prog.progress((i+1)/len(target_exps))
    prog.empty()
    return S, pd.concat(dfs) if dfs else None

def process_exposure(df, S, s_range):
    if df is None or df.empty: return pd.DataFrame()
    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range)].copy()
    res = []
    for _, row in df.iterrows():
        g = row.get('greeks')
        if not g or not isinstance(g, dict): continue
        gamma, vanna, oi = float(g.get('gamma', 0) or 0), float(g.get('vanna', 0) or 0), int(row.get('open_interest', 0) or 0)
        # Dealer position logic: Short Calls (-), Long Puts (+)
        dealer_pos = -1 if row['option_type'].lower() == 'call' else 1
        res.append({
            "strike": row['strike'], "expiry": row['expiration_date'],
            "gex": dealer_pos * gamma * (S**2) * 0.01 * CONTRACT_SIZE * oi,
            "vex": dealer_pos * vanna * CONTRACT_SIZE * oi
        })
    return pd.DataFrame(res)

def render_plots(df, ticker, S, mode):
    if df.empty: return None, None
    val_col = mode.lower()
    agg = df.groupby('strike')[val_col].sum().sort_index()
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_col, aggfunc='sum', fill_value=0).sort_index(ascending=False)

    z_raw = pivot.values
    x_labs, y_labs = pivot.columns.tolist(), pivot.index.tolist()
    
    max_abs = np.max(np.abs(z_raw)) if z_raw.size else 1.0
    fig_h = go.Figure(data=go.Heatmap(
        z=z_raw, x=x_labs, y=y_labs, colorscale='Viridis', zmid=0, zmin=-max_abs, zmax=max_abs,
        colorbar=dict(title=f"{mode} ($)")
    ))

    # Conditional Annotations: Positive=Black, Negative=White
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < 500: continue
            font_color = "black" if val >= 0 else "white"
            fig_h.add_annotation(
                x=exp, y=strike, text=f"${abs(val)/1e3:,.0f}K",
                showarrow=False, font=dict(color=font_color, size=10, family="Arial Black")
            )

    fig_h.update_layout(title=f"{ticker} {mode} | Spot: ${S:,.2f}", template="plotly_dark", height=850)
    fig_b = go.Figure(go.Bar(x=agg.index, y=agg.values, marker_color=['#2563eb' if v < 0 else '#fbbf24' for v in agg.values]))
    fig_b.update_layout(title=f"Net {mode} by Strike", template="plotly_dark", height=350)
    return fig_h, fig_b

# -------------------------
# Main App
# -------------------------
def main():
    st.markdown("<div style='text-align:center;'><h2 style='font-size:18px;'>📊 GEX / VEX Pro (Tradier Native)</h2></div>", unsafe_allow_html=True)
    
    ticker = st.text_input("Ticker", "SPX").upper().strip()
    
    c1, c2, c3, c4 = st.columns([1, 1, 1, 0.8])
    with c1: mode = st.radio("Metric", ["GEX", "VEX"], horizontal=True)
    with c2: max_exp = st.number_input("Expiries", 1, 15, 6)
    with c3: s_range = st.number_input("Strike ±", 5, 1000, 80 if ticker == "SPX" else 30)
    with c4: run = st.button("Run", type="primary")

    if run:
        S, raw_df = fetch_tradier_data(ticker, int(max_exp))
        if S and raw_df is not None:
            processed = process_exposure(raw_df, S, s_range)
            if not processed.empty:
                net_val = processed[mode.lower()].sum() / 1e9
                st.metric(f"Total Net {mode}", f"${net_val:,.2f}B")
                h_fig, b_fig = render_plots(processed, ticker, S, mode)
                st.plotly_chart(h_fig, use_container_width=True)
                st.plotly_chart(b_fig, use_container_width=True)
            else: st.warning("No data found in range.")
        else: st.error("Fetch failed. Verify symbol and API Token.")

if __name__ == "__main__":
    main()