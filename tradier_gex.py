import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
from datetime import datetime

# --- CREDENTIALS LOADING ---
# Completely bypass token box if Secret exists
if "TRADIER_TOKEN" in st.secrets and st.secrets["TRADIER_TOKEN"]:
    saved_token = st.secrets["TRADIER_TOKEN"]
    show_token_box = False
else:
    saved_token = ""
    show_token_box = True

# --- APP CONFIG ---
st.set_page_config(page_title="GEX Pro Live", page_icon="⚡", layout="wide")

BASE_URL = "https://api.tradier.com/v1/"
CONTRACT_SIZE = 100

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

@st.cache_data(ttl=3600)
def fetch_trading_calendar(token):
    """Fetches real-time market calendar from Tradier."""
    data = get_tradier_data("markets/calendar", {}, token)
    open_dates = set()
    try:
        if data and 'calendar' in data:
            days = data['calendar']['days']['day']
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
        if not g: continue
        gamma, vanna, oi = (g.get('gamma', 0) or 0), (g.get('vanna', 0) or 0), (row.get('open_interest', 0) or 0)
        is_call = row['option_type'].lower() == 'call'
        # Standard Dealer: Short Calls (-), Long Puts (+)
        gex = (-gamma if is_call else gamma) * S**2 * 0.01 * CONTRACT_SIZE * oi
        vex = (-vanna if is_call else vanna) * CONTRACT_SIZE * oi
        res.append({"strike": row['strike'], "expiry": row['expiration_date'], "gex": gex, "vex": vex})
    return pd.DataFrame(res)

def calculate_gamma_flip(df):
    agg = df.groupby('strike')['gex'].sum().sort_index()
    s, v = agg.index.values, agg.values
    for i in range(len(v) - 1):
        if (v[i] < 0 and v[i+1] > 0) or (v[i] > 0 and v[i+1] < 0):
            return s[i] - v[i] * (s[i+1] - s[i]) / (v[i+1] - v[i])
    return None

def render_plots(df, ticker, S, mode):
    val_col = mode.lower()
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_col, aggfunc='sum').sort_index(ascending=False)
    z_raw, y_labs, x_labs = pivot.values, pivot.index.tolist(), pivot.columns.tolist()
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** 0.5)
    closest_strike = min(y_labs, key=lambda x: abs(x - S))
    
    fig = go.Figure(data=go.Heatmap(z=z_scaled, x=x_labs, y=y_labs, colorscale='Viridis', zmid=0, showscale=True, text=z_raw, hoverinfo="none"))
    strike_diff = np.mean(np.diff(sorted(y_labs))) if len(y_labs) > 1 else 5
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < 500: continue
            fig.add_annotation(x=exp, y=strike, text=f"{'-' if val < 0 else ''}${abs(val)/1e3:,.0f}K", showarrow=False, font=dict(color="white", size=10))

    fig.add_shape(type="rect", xref="paper", yref="y", x0=-0.05, x1=1.05, y0=closest_strike-(strike_diff*0.4), y1=closest_strike+(strike_diff*0.4), fillcolor="rgba(255, 0, 0, 0.25)", line=dict(width=0), layer="below")
    fig.update_layout(title=f"LIVE {ticker} {mode} | Spot: ${S:,.2f}", template="plotly_dark", height=750, margin=dict(t=50, b=50), yaxis=dict(tickmode='array', tickvals=y_labs, ticktext=[f"<b>{s:,.0f}</b>" if s == closest_strike else f"{s:,.0f}" for s in y_labs]))
    return fig

# -------------------------
# MAIN APP
# -------------------------
st.title("📊 GEX & VEX Pro Live")

if show_token_box:
    api_token = st.text_input("Enter Tradier Token", type="password")
else:
    api_token = saved_token

# Dynamic Default Strikes logic
ticker_input = st.text_input("Ticker", "SPX").upper().strip()
default_range = 80 if "SPX" in ticker_input else 25

c1, c2, c3 = st.columns([1, 1, 3])
with c1: mode = st.selectbox("Metric", ["GEX", "VEX"])
with c2: max_exp = st.number_input("Expiries", 1, 15, 5)
with c3: s_range = st.slider("Strike Range (± Spot)", 5, 250, default_range)

@st.fragment(run_every="60s")
def live_pulse():
    if not api_token:
        st.info("Please set TRADIER_TOKEN in Streamlit Secrets.")
        return

    open_dates = fetch_trading_calendar(api_token)
    S = fetch_market_data(ticker_input, api_token)
    exp_data = get_tradier_data("markets/options/expirations", {"symbol": ticker_input}, api_token)
    
    if S and exp_data:
        prog_bar = st.progress(0, text="Fetching chains...")
        raw_dates = exp_data['expirations']['date']
        if not isinstance(raw_dates, list): raw_dates = [raw_dates]
        
        # Filter for Market-Open days only
        valid_dates = [d for d in raw_dates if d in open_dates]
        target_dates = valid_dates[:max_exp]
        
        all_chains = []
        for i, d in enumerate(target_dates):
            all_chains.append(fetch_options_chain(ticker_input, d, api_token))
            prog_bar.progress((i + 1) / len(target_dates))
        
        prog_bar.empty()
        full_df = pd.concat(all_chains)
        processed = process_exposure(full_df, S, s_range)
        
        if not processed.empty:
            net_val = processed[mode.lower()].sum() / 1e9
            st.metric(f"Total Net {mode}", f"{'-' if net_val < 0 else ''}${abs(net_val):,.2f}B")
            st.plotly_chart(render_plots(processed, ticker_input, S, mode), use_container_width=True)
            
            st.markdown("---")
            st.subheader("🧱 Gamma Wall Analysis")
            agg_gex = processed.groupby('strike')['gex'].sum()
            flip_price = calculate_gamma_flip(processed)
            w1, w2, w3 = st.columns(3)
            w1.metric("Call Wall", f"${agg_gex.idxmax():,.0f}")
            w2.metric("Put Wall", f"${agg_gex.idxmin():,.0f}")
            w3.metric("Gamma Flip", f"${flip_price:,.2f}" if flip_price else "N/A")
            st.caption(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")
    else: st.error("Connection failed. Check Ticker.")

live_pulse()