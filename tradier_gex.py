import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
from datetime import datetime

# --- CREDENTIALS LOADING ---
# Completely remove token box if Secret exists
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
# Tradier API & Date Logic
# -------------------------
def get_tradier_data(endpoint, params, token):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers)
        if response.status_code == 200: return response.json()
    except: return None
    return None

def is_valid_trading_day(date_str, open_dates_from_api):
    """Combines API calendar with a manual holiday/weekend override."""
    dt = pd.to_datetime(date_str)
    # 1. Block Weekends
    if dt.weekday() > 4: return False
    # 2. Block Known Holidays (Dec 25, Jan 1, etc.)
    hard_holidays = ['2025-12-25', '2026-01-01', '2026-01-19']
    if date_str in hard_holidays: return False
    # 3. Cross-reference with Tradier Calendar
    if open_dates_from_api and date_str not in open_dates_from_api:
        return False
    return True

@st.cache_data(ttl=3600)
def fetch_trading_calendar(token):
    data = get_tradier_data("markets/calendar", {}, token)
    open_dates = set()
    try:
        if data and 'calendar' in data:
            days = data['calendar']['days']['day']
            if isinstance(days, dict): days = [days]
            for d in days:
                if d['status'] == 'open': open_dates.add(d['date'])
    except: pass
    return open_dates

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
        # Long Puts (+), Short Calls (-)
        gex = (-gamma if is_call else gamma) * S**2 * 0.01 * CONTRACT_SIZE * oi
        vex = (-vanna if is_call else vanna) * CONTRACT_SIZE * oi
        res.append({"strike": row['strike'], "expiry": row['expiration_date'], "gex": gex, "vex": vex})
    return pd.DataFrame(res)

# -------------------------
# MAIN UI
# -------------------------
st.title("📊 GEX & VEX Pro Live")

if show_token_box:
    api_token = st.text_input("Enter Tradier Token", type="password")

ticker = st.text_input("Ticker", "SPX").upper().strip()
default_range = 80 if ticker == "SPX" else 25

c1, c2, c3 = st.columns([1, 1, 3])
with c1: mode = st.selectbox("Metric", ["GEX", "VEX"])
with c2: max_exp = st.number_input("Expiries", 1, 15, 5)
with c3: s_range = st.slider("Strike Range (± Spot)", 5, 250, default_range)

@st.fragment(run_every="60s")
def live_pulse():
    if not api_token:
        st.info("Set TRADIER_TOKEN in Streamlit Secrets.")
        return

    open_dates = fetch_trading_calendar(api_token)
    S = fetch_market_data(ticker, api_token)
    exp_data = get_tradier_data("markets/options/expirations", {"symbol": ticker}, api_token)
    
    if S and exp_data:
        prog_bar = st.progress(0, text="Syncing Market Data...")
        raw_dates = exp_data['expirations']['date']
        if not isinstance(raw_dates, list): raw_dates = [raw_dates]
        
        # FILTER: Exclude weekends and Christmas/Holidays
        target_dates = [d for d in raw_dates if is_valid_trading_day(d, open_dates)][:max_exp]
        
        all_chains = []
        for i, d in enumerate(target_dates):
            df = fetch_options_chain(ticker, d, api_token)
            if not df.empty: all_chains.append(df)
            prog_bar.progress((i + 1) / len(target_dates))
        
        prog_bar.empty()
        
        if not all_chains:
            st.warning(f"No active trading data for {ticker} on upcoming open days.")
            return

        full_df = pd.concat(all_chains)
        processed = process_exposure(full_df, S, s_range)
        
        if not processed.empty:
            # Stats & Heatmap
            net_val = processed[mode.lower()].sum() / 1e9
            st.metric(f"Total Net {mode}", f"{'-' if net_val < 0 else ''}${abs(net_val):,.2f}B")
            
            # Pivot & Render Plotly
            pivot = processed.pivot_table(index='strike', columns='expiry', values=mode.lower(), aggfunc='sum').sort_index(ascending=False)
            z_raw = pivot.values
            z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** 0.5)
            
            fig = go.Figure(data=go.Heatmap(z=z_scaled, x=pivot.columns, y=pivot.index, colorscale='Viridis', zmid=0, text=z_raw, hoverinfo="none"))
            fig.update_layout(title=f"LIVE {ticker} {mode} | Spot: ${S:,.2f}", template="plotly_dark", height=750)
            st.plotly_chart(fig, use_container_width=True)
            
            # Walls
            st.markdown("---")
            agg_gex = processed.groupby('strike')['gex'].sum()
            w1, w2 = st.columns(2)
            w1.metric("Call Wall (+GEX)", f"${agg_gex.idxmax():,.0f}")
            w2.metric("Put Wall (-GEX)", f"${agg_gex.idxmin():,.0f}")
            st.caption(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")
    else: st.error(f"Data Fetch Failed for {ticker}.")

live_pulse()