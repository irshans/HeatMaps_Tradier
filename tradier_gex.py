import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
from datetime import datetime
from zoneinfo import ZoneInfo

# --- APP CONFIG ---
st.set_page_config(page_title="GEX Pro 2025", page_icon="📊", layout="wide")

# --- CREDENTIALS ---
# Uses Streamlit Secrets for the Tradier Token
if "TRADIER_TOKEN" in st.secrets:
    TRADIER_TOKEN = st.secrets["TRADIER_TOKEN"]
else:
    st.error("Please set TRADIER_TOKEN in your Streamlit Secrets.")
    st.stop()

BASE_URL = "https://api.tradier.com/v1/"
CONTRACT_SIZE = 100

# Compact UI styling
st.markdown(
    """
    <style>
    .block-container { padding-top: 24px; padding-bottom: 8px; }
    button[kind="primary"], .stButton>button { padding:4px 8px !important; font-size:12px !important; height:30px !important; }
    input[type="text"], input[type="number"], select { padding:6px 8px !important; font-size:12px !important; height:28px !important; }
    div[role="radiogroup"] label, .stSelectbox, .stRadio { font-size:12px !important; }
    .stSlider > div, .stNumberInput > div { font-size:12px !important; height:34px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Tradier API Functions
# -------------------------
def tradier_get(endpoint, params):
    headers = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Tradier API Error: {e}")
    return None

def fetch_tradier_data(ticker, max_exp):
    # 1. Get Spot Price
    quote_data = tradier_get("markets/quotes", {"symbols": ticker})
    if not quote_data or 'quotes' not in quote_data:
        return None, None
    
    quote = quote_data['quotes']['quote']
    S = float(quote['last']) if isinstance(quote, dict) else float(quote[0]['last'])

    # 2. Get Expirations
    exp_data = tradier_get("markets/options/expirations", {"symbol": ticker, "includeAllRoots": "true"})
    if not exp_data or 'expirations' not in exp_data:
        return S, None
    
    all_exps = exp_data['expirations']['date']
    if not isinstance(all_exps, list): all_exps = [all_exps]
    target_exps = all_exps[:max_exp]

    # 3. Fetch Chains with Greeks
    dfs = []
    prog_bar = st.progress(0)
    for i, exp in enumerate(target_exps):
        chain_data = tradier_get("markets/options/chains", {"symbol": ticker, "expiration": exp, "greeks": "true"})
        if chain_data and 'options' in chain_data and chain_data['options']:
            opts = chain_data['options']['option']
            df = pd.DataFrame(opts) if isinstance(opts, list) else pd.DataFrame([opts])
            dfs.append(df)
        prog_bar.progress((i + 1) / len(target_exps))
    
    prog_bar.empty()
    if not dfs: return S, None
    
    full_df = pd.concat(dfs, ignore_index=True)
    return S, full_df

# -------------------------
# Processing logic (Dealer Model: Short Calls / Long Puts)
# -------------------------
def process_exposure(df, S, s_range):
    if df is None or df.empty: return pd.DataFrame()
    
    # Filter by Strike Range
    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range)].copy()
    
    res = []
    for _, row in df.iterrows():
        g = row.get('greeks')
        if not g: continue
        
        # Tradier Greeks are per share, so multiply by S^2 * 0.01 for 1% GEX
        gamma = float(g.get('gamma', 0) or 0)
        vanna = float(g.get('vanna', 0) or 0)
        oi = int(row.get('open_interest', 0) or 0)
        
        # Dealer Position Logic
        is_call = row['option_type'].lower() == 'call'
        dealer_pos = -1 if is_call else 1
        
        # GEX: Dealer Pos * Gamma * Spot^2 * 0.01 * 100 shares * OI
        gex = dealer_pos * gamma * (S ** 2) * 0.01 * CONTRACT_SIZE * oi
        # VEX: Dealer Pos * Vanna * 100 shares * OI (Vanna is dDelta/dVol)
        vex = dealer_pos * vanna * CONTRACT_SIZE * oi
        
        res.append({
            "strike": row['strike'],
            "expiry": row['expiration_date'],
            "gex": gex,
            "vex": vex
        })
        
    return pd.DataFrame(res)

# -------------------------
# Plots
# -------------------------
def render_plots(df, ticker, S, mode):
    if df.empty: return None, None
    val_col = mode.lower()
    agg = df.groupby('strike')[val_col].sum().sort_index()
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_col, aggfunc='sum', fill_value=0).sort_index(ascending=False)

    z_raw = pivot.values
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** 0.5)
    x_labs, y_labs = pivot.columns.tolist(), pivot.index.tolist()
    closest_strike = min(y_labs, key=lambda x: abs(x - S))

    # Build Heatmap
    max_abs = np.max(np.abs(z_raw)) if z_raw.size else 1.0
    fig_h = go.Figure(data=go.Heatmap(
        z=z_raw, x=x_labs, y=y_labs, colorscale='Viridis', zmid=0, zmin=-max_abs, zmax=max_abs,
        colorbar=dict(title=f"{mode} ($)")
    ))

    # Add annotations
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < 500: continue
            fig_h.add_annotation(x=exp, y=strike, text=f"${abs(val)/1e3:,.0f}K", showarrow=False, font=dict(color="white", size=10))

    fig_h.update_layout(title=f"{ticker} {mode} Exposure Map | Spot: ${S:,.2f}", template="plotly_dark", height=800)
    
    fig_b = go.Figure(go.Bar(x=agg.index, y=agg.values, marker_color=['#2563eb' if v < 0 else '#fbbf24' for v in agg.values]))
    fig_b.update_layout(title=f"Net {mode} by Strike", template="plotly_dark", height=350)
    
    return fig_h, fig_b

# -------------------------
# Main App
# -------------------------
def main():
    st.markdown("<div style='text-align:center;'><h2 style='font-size:18px;'>📊 GEX / VEX Pro (Tradier Live)</h2></div>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns([1.5, 1, 0.8, 1, 0.8])
    with col1: ticker = st.text_input("Ticker", "SPY").upper().strip()
    with col2: mode = st.radio("Metric", ["GEX", "VEX"], horizontal=True)
    with col3: max_exp = st.number_input("Expiries", 1, 15, 6)
    with col4: s_range = st.number_input("Strike ±", 5, 500, 30)
    with col5: run = st.button("Run", type="primary")

    if run:
        with st.spinner(f"Fetching Tradier Data for {ticker}..."):
            S, raw_df = fetch_tradier_data(ticker, int(max_exp))

        if S and raw_df is not None:
            processed = process_exposure(raw_df, S, s_range)
            if not processed.empty:
                t_gex, t_vex = processed["gex"].sum() / 1e9, processed["vex"].sum() / 1e9
                m1, m2 = st.columns(2)
                m1.metric("Net Dealer GEX", f"${t_gex:,.2f}B")
                m2.metric("Net Dealer VEX", f"${t_vex:,.2f}B")

                h_fig, b_fig = render_plots(processed, ticker, S, mode)
                st.plotly_chart(h_fig, use_container_width=True)
                st.plotly_chart(b_fig, use_container_width=True)
            else:
                st.warning("No data in range.")
        else:
            st.error("Fetch failed. Verify ticker and API Token.")

if __name__ == "__main__":
    main()