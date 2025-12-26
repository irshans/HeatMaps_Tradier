import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
from datetime import datetime

# --- CREDENTIALS LOADING ---
if "TRADIER_TOKEN" in st.secrets:
    saved_token = st.secrets["TRADIER_TOKEN"]
else:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        saved_token = os.getenv("TRADIER_TOKEN", "")
    except:
        saved_token = ""

# --- APP CONFIG ---
st.set_page_config(page_title="GEX Live Pulse", page_icon="⚡", layout="wide")

BASE_URL = "https://api.tradier.com/v1/"
CONTRACT_SIZE = 100

# -------------------------
# Tradier API Functions
# -------------------------
def get_tradier_data(endpoint, params, token):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers)
        if response.status_code == 200:
            return response.json()
    except:
        return None
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
        if not greeks or greeks is None: continue
        gamma = greeks.get('gamma', 0) or 0
        delta = greeks.get('delta', 0) or 0
        oi = row.get('open_interest', 0) or 0
        if model_type == "Dealer Short All":
            gex = -gamma * S**2 * 0.01 * CONTRACT_SIZE * oi
        else:
            gex = (-gamma if row['option_type'] == 'call' else gamma) * S**2 * 0.01 * CONTRACT_SIZE * oi
        dex = -delta * S * CONTRACT_SIZE * oi
        res.append({"strike": row['strike'], "expiry": row['expiration_date'], "gex": gex, "dex": dex})
    return pd.DataFrame(res)

# -------------------------
# Plotting
# -------------------------
def render_plots(df, ticker, S, mode):
    val_col = mode.lower()
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_col, aggfunc='sum').sort_index(ascending=False)
    z_raw = pivot.values
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** 0.5)
    y_labs, x_labs = pivot.index.tolist(), pivot.columns.tolist()
    closest_strike = min(y_labs, key=lambda x: abs(x - S))

    fig = go.Figure(data=go.Heatmap(
        z=z_scaled, x=x_labs, y=y_labs, colorscale='Viridis', zmid=0, showscale=True,
        text=z_raw, hoverinfo="none"
    ))

    strike_diff = np.mean(np.diff(sorted(y_labs))) if len(y_labs) > 1 else 5
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < 500: continue
            txt = f"{'-' if val < 0 else ''}${abs(val)/1e3:,.0f}K"
            fig.add_annotation(x=exp, y=strike, text=txt, showarrow=False, font=dict(color="white", size=10))

    fig.add_shape(type="rect", xref="paper", yref="y", x0=-0.05, x1=1.05, 
                  y0=closest_strike-(strike_diff*0.4), y1=closest_strike+(strike_diff*0.4),
                  fillcolor="rgba(255, 0, 0, 0.25)", line=dict(width=0), layer="below")

    fig.update_layout(
        title=f"LIVE {ticker} {mode} | Spot: ${S:,.2f}",
        template="plotly_dark", height=800, margin=dict(t=50, b=50),
        yaxis=dict(tickmode='array', tickvals=y_labs, 
                   ticktext=[f"<b>{s:,.0f}</b>" if s == closest_strike else f"{s:,.0f}" for s in y_labs])
    )
    return fig

# -------------------------
# MAIN APP
# -------------------------
st.title("📊 GEX Pro Live")

# TOP CONTROLS
c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1, 1.5])
with c1:
    api_token = st.text_input("Tradier Token", value=saved_token, type="password")
with c2:
    ticker = st.text_input("Ticker", "SPY").upper().strip()
with c3:
    mode = st.selectbox("Metric", ["GEX", "DEX"])
with c4:
    max_exp = st.number_input("Expiries", 1, 10, 5)
with c5:
    model_type = st.selectbox("Model", ["Standard", "Dealer Short All"])

s_range = st.slider("Strike Range (± Spot)", 5, 200, 40)

# LIVE REFRESH FRAGMENT
@st.fragment(run_every="60s")
def live_pulse():
    if not api_token:
        st.info("Please enter a Tradier Token to begin.")
        return

    S = fetch_market_data(ticker, api_token)
    exp_data = get_tradier_data("markets/options/expirations", {"symbol": ticker}, api_token)
    
    if S and exp_data:
        dates = exp_data['expirations']['date']
        if not isinstance(dates, list): dates = [dates]
        
        all_chains = [fetch_options_chain(ticker, d, api_token) for d in dates[:max_exp]]
        full_df = pd.concat(all_chains)
        processed = process_exposure(full_df, S, s_range, model_type)
        
        if not processed.empty:
            net_val = processed[mode.lower()].sum() / 1e9
            st.metric(f"Total Net {mode}", f"{'-' if net_val < 0 else ''}${abs(net_val):,.2f}B", 
                      help="Updated automatically every 60 seconds")
            
            fig = render_plots(processed, ticker, S, mode)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.warning("No data in range.")
    else:
        st.error("Connection failed. Verify Token and Ticker.")

live_pulse()