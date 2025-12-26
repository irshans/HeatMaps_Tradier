import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# --- LOAD SAVED TOKEN ---
load_dotenv()
saved_token = os.getenv("TRADIER_TOKEN", "")

# --- APP CONFIG ---
st.set_page_config(page_title="Tradier Live Pulse", page_icon="⚡", layout="wide")

BASE_URL = "https://api.tradier.com/v1/"
CONTRACT_SIZE = 100

# -------------------------
# Tradier Data Fetchers
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
        return pd.DataFrame(opts) if isinstance(opts, list) else pd.DataFrame([opts])
    return pd.DataFrame()

# -------------------------
# Processing Logic
# -------------------------
def process_exposure(df, S, s_range, model_type):
    if df.empty: return pd.DataFrame()
    df = df[(df['strike'] >= S - s_range) & (df['strike'] <= S + s_range)].copy()
    res = []
    for _, row in df.iterrows():
        greeks = row.get('greeks', {})
        if not greeks: continue
        gamma, delta = (greeks.get('gamma', 0) or 0), (greeks.get('delta', 0) or 0)
        oi = row.get('open_interest', 0) or 0
        if model_type == "Dealer Short All (Absolute Stress)":
            gex = -gamma * S**2 * 0.01 * CONTRACT_SIZE * oi
        else:
            gex = (-gamma if row['option_type'] == 'call' else gamma) * S**2 * 0.01 * CONTRACT_SIZE * oi
        dex = -delta * S * CONTRACT_SIZE * oi
        res.append({"strike": row['strike'], "expiry": row['expiration'], "gex": gex, "dex": dex})
    return pd.DataFrame(res)

# -------------------------
# Visualizations
# -------------------------
def render_plots(df, ticker, S, mode):
    val_col = mode.lower()
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_col, aggfunc='sum').sort_index(ascending=False)
    z_raw = pivot.values
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** 0.5)
    y_labs, x_labs = pivot.index.tolist(), pivot.columns.tolist()
    closest_strike = min(y_labs, key=lambda x: abs(x - S))

    fig_h = go.Figure(data=go.Heatmap(
        z=z_scaled, x=x_labs, y=y_labs, 
        colorscale='Viridis', zmid=0, showscale=True,
        text=z_raw, hoverinfo="none"
    ))

    # Cell annotations & Red Spot Highlight
    strike_diff = np.mean(np.diff(sorted(y_labs))) if len(y_labs) > 1 else 5
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < 500: continue
            txt = f"{'-' if val < 0 else ''}${abs(val)/1e3:,.0f}K"
            fig_h.add_annotation(x=exp, y=strike, text=txt, showarrow=False, font=dict(color="white", size=11))

    fig_h.add_shape(type="rect", xref="paper", yref="y", x0=-0.05, x1=1.05, 
                    y0=closest_strike-(strike_diff*0.4), y1=closest_strike+(strike_diff*0.4),
                    fillcolor="rgba(255, 0, 0, 0.2)", line=dict(width=0), layer="below")

    fig_h.update_layout(title=f"LIVE {ticker} {mode} | Last Spot: ${S:,.2f}", template="plotly_dark", height=800)
    return fig_h

# -------------------------
# LIVE FRAGMENT COMPONENT
# -------------------------
@st.fragment(run_every="60s")
def live_data_fragment(ticker, api_token, max_exp, s_range, model_type, mode):
    S = fetch_market_data(ticker, api_token)
    exp_data = get_tradier_data("markets/options/expirations", {"symbol": ticker}, api_token)
    
    if S and exp_data:
        dates = exp_data['expirations']['date']
        if not isinstance(dates, list): dates = [dates]
        
        all_chains = [fetch_options_chain(ticker, d, api_token) for d in dates[:max_exp]]
        full_df = pd.concat(all_chains)
        processed = process_exposure(full_df, S, s_range, model_type)
        
        if not processed.empty:
            val = processed[mode.lower()].sum() / 1e9
            st.metric(f"Total Net {mode} (Auto-updates every 60s)", f"{'-' if val < 0 else ''}${abs(val):,.2f}B")
            st.plotly_chart(render_plots(processed, ticker, S, mode), use_container_width=True)
            st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    else:
        st.error("Waiting for data...")

# -------------------------
# Main App
# -------------------------
def main():
    st.sidebar.title("⚡ Live Pulse GEX")
    api_token = st.sidebar.text_input("Bearer Token", value=saved_token, type="password")
    ticker = st.sidebar.text_input("Ticker", "SPY").upper().strip()
    mode = st.sidebar.radio("Metric", ["GEX", "DEX"])
    model_type = st.sidebar.selectbox("Model", ["Standard", "Dealer Short All"])
    max_exp = st.sidebar.slider("Expirations", 1, 10, 5)
    s_range = st.sidebar.slider("Strike Range", 5, 100, 30)

    if api_token:
        live_data_fragment(ticker, api_token, max_exp, s_range, model_type, mode)
    else:
        st.info("Please enter your Tradier Token in the sidebar to start.")

if __name__ == "__main__":
    main()