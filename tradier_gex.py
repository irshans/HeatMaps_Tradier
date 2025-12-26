import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
from datetime import datetime

# --- CREDENTIALS LOADING ---
# If TRADIER_TOKEN is in secrets, we skip the text input entirely.
if "TRADIER_TOKEN" in st.secrets and st.secrets["TRADIER_TOKEN"]:
    api_token = st.secrets["TRADIER_TOKEN"]
    show_token_box = False
else:
    api_token = ""
    show_token_box = True

# --- APP CONFIG ---
st.set_page_config(page_title="GEX Pro Live", page_icon="⚡", layout="wide")

BASE_URL = "https://api.tradier.com/v1/"
CONTRACT_SIZE = 100

# -------------------------
# Date Filtering Logic
# -------------------------
def is_valid_trading_day(date_str):
    """Exclude weekends and specific major holidays."""
    dt = pd.to_datetime(date_str)
    # 1. Block Weekends (Saturday=5, Sunday=6)
    if dt.weekday() > 4: 
        return False
    # 2. Block Christmas & New Years
    hard_holidays = ['2025-12-25', '2026-01-01']
    if date_str in hard_holidays:
        return False
    return True

# -------------------------
# Tradier API Functions
# -------------------------
def get_tradier_data(endpoint, params, token):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers)
        if response.status_code == 200: return response.json()
    except: return None
    return None

def fetch_market_data(ticker, token):
    data = get_tradier_data("markets/quotes", {"symbols": ticker}, token)
    if data and 'quotes' in data and 'quote' in data['quotes']:
        quote = data['quotes']['quote']
        return quote['last'] if isinstance(quote, dict) else quote[0]['last']
    return None

def fetch_options_chain(ticker, expiry, token):
    params = {"symbol": ticker, "expiration": expiry, "greeks": "true"}
    data = get_tradier_data("markets/options/chains", params, token)
    if data and 'options' in data and data['options']:
        opts = data['options']['option']
        if opts is None: return pd.DataFrame()
        return pd.DataFrame(opts) if isinstance(opts, list) else pd.DataFrame([opts])
    return pd.DataFrame()

# -------------------------
# Processing Logic
# -------------------------
def process_exposure(df, S, s_range):
    if df.empty: return pd.DataFrame()
    df = df[(df['strike'] >= S - s_range) & (df['strike'] <= S + s_range)].copy()
    res = []
    for _, row in df.iterrows():
        g = row.get('greeks', {})
        if not g or not isinstance(g, dict): continue
        gamma, vanna, oi = (g.get('gamma', 0) or 0), (g.get('vanna', 0) or 0), (row.get('open_interest', 0) or 0)
        is_call = row['option_type'].lower() == 'call'
        # Long Puts (+GEX), Short Calls (-GEX)
        gex = (-gamma if is_call else gamma) * S**2 * 0.01 * CONTRACT_SIZE * oi
        vex = (-vanna if is_call else vanna) * CONTRACT_SIZE * oi
        res.append({"strike": row['strike'], "expiry": row['expiration_date'], "gex": gex, "vex": vex})
    return pd.DataFrame(res)

# -------------------------
# MAIN UI
# -------------------------
st.title("📊 GEX & VEX Pro Live")

# The token box ONLY appears if the secret is missing
if show_token_box:
    api_token = st.sidebar.text_input("Enter Tradier Token", type="password")

ticker = st.text_input("Ticker", "SPX").upper().strip()

# Adjust default strike range based on ticker
default_range = 80 if ticker == "SPX" else 25

c1, c2, c3 = st.columns([1, 1, 3])
with c1: mode = st.selectbox("Metric", ["GEX", "VEX"])
with c2: max_exp = st.number_input("Expiries", 1, 15, 5)
with c3: s_range = st.slider("Strike Range (± Spot)", 5, 250, default_range)

@st.fragment(run_every="60s")
def live_pulse():
    if not api_token:
        st.info("Please set TRADIER_TOKEN in Streamlit Secrets.")
        return

    S = fetch_market_data(ticker, api_token)
    exp_data = get_tradier_data("markets/options/expirations", {"symbol": ticker}, api_token)
    
    if S and exp_data:
        prog_bar = st.progress(0, text=f"Syncing {ticker} Chains...")
        
        raw_dates = exp_data['expirations']['date']
        if not isinstance(raw_dates, list): raw_dates = [raw_dates]
        
        # Filter dates to ensure only valid trading days are fetched
        target_dates = [d for d in raw_dates if is_valid_trading_day(d)][:max_exp]
        
        all_chains = []
        for i, d in enumerate(target_dates):
            df = fetch_options_chain(ticker, d, api_token)
            if not df.empty: all_chains.append(df)
            prog_bar.progress((i + 1) / len(target_dates))
        
        prog_bar.empty()
        
        if not all_chains:
            st.warning(f"No valid trading data for {ticker} on upcoming dates.")
            return

        full_df = pd.concat(all_chains)
        processed = process_exposure(full_df, S, s_range)
        
        if not processed.empty:
            # Main Metric
            net_val = processed[mode.lower()].sum() / 1e9
            st.metric(f"Total Net {mode}", f"{'-' if net_val < 0 else ''}${abs(net_val):,.2f}B")
            
            # Pivot & Map
            pivot = processed.pivot_table(index='strike', columns='expiry', values=mode.lower(), aggfunc='sum').sort_index(ascending=False)
            z_raw = pivot.values
            z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** 0.5) # Scale for color visibility
            
            fig = go.Figure(data=go.Heatmap(
                z=z_scaled, x=pivot.columns, y=pivot.index, 
                colorscale='Viridis', zmid=0, text=z_raw, hoverinfo="none"
            ))
            fig.update_layout(title=f"LIVE {ticker} {mode} | Spot: ${S:,.2f}", template="plotly_dark", height=750)
            st.plotly_chart(fig, use_container_width=True)
            
            # Wall Stats
            st.markdown("---")
            agg_gex = processed.groupby('strike')['gex'].sum()
            w1, w2 = st.columns(2)
            w1.metric("Call Wall (Max +GEX)", f"${agg_gex.idxmax():,.0f}")
            w2.metric("Put Wall (Max -GEX)", f"${agg_gex.idxmin():,.0f}")
            st.caption(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.error("Strike range returned no data. Try increasing the range slider.")
    else:
        st.error(f"Failed to find symbol: {ticker}")

live_pulse()