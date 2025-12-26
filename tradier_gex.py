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
st.set_page_config(page_title="GEX & VEX Pro Live", page_icon="⚡", layout="wide")

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
        vanna = greeks.get('vanna', 0) or 0
        oi = row.get('open_interest', 0) or 0
        
        if model_type == "Dealer Short All":
            gex = -gamma * S**2 * 0.01 * CONTRACT_SIZE * oi
            vex = -vanna * CONTRACT_SIZE * oi 
        else:
            gex = (-gamma if row['option_type'] == 'call' else gamma) * S**2 * 0.01 * CONTRACT_SIZE * oi
            vex = (-vanna if row['option_type'] == 'call' else vanna) * CONTRACT_SIZE * oi

        res.append({
            "strike": row['strike'], 
            "expiry": row['expiration_date'], 
            "gex": gex, 
            "vex": vex
        })
    return pd.DataFrame(res)

def calculate_gamma_flip(df):
    """Simple linear interpolation to find where GEX crosses zero."""
    agg = df.groupby('strike')['gex'].sum().sort_index()
    strikes = agg.index.values
    values = agg.values
    for i in range(len(values) - 1):
        if (values[i] < 0 and values[i+1] > 0) or (values[i] > 0 and values[i+1] < 0):
            # Linear interpolation
            low_s, high_s = strikes[i], strikes[i+1]
            low_v, high_v = values[i], values[i+1]
            flip = low_s - low_v * (high_s - low_s) / (high_v - low_v)
            return flip
    return None

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
        template="plotly_dark", height=700, margin=dict(t=50, b=50),
        yaxis=dict(tickmode='array', tickvals=y_labs, 
                   ticktext=[f"<b>{s:,.0f}</b>" if s == closest_strike else f"{s:,.0f}" for s in y_labs])
    )
    return fig

# -------------------------
# MAIN APP
# -------------------------
st.title("📊 GEX & VEX Pro Live")

if not saved_token:
    api_token = st.text_input("Enter Tradier Token", type="password")
else:
    api_token = saved_token

c1, c2, c3, c4 = st.columns([1, 1, 1, 1.5])
with c1:
    ticker = st.text_input("Ticker", "SPY").upper().strip()
with c2:
    mode = st.selectbox("Metric", ["GEX", "VEX"])
with c3:
    max_exp = st.number_input("Expiries", 1, 10, 3)
with c4:
    model_type = st.selectbox("Dealer Model", ["Standard", "Dealer Short All"])

s_range = st.slider("Strike Range (± Spot)", 5, 200, 40)

@st.fragment(run_every="60s")
def live_pulse():
    if not api_token:
        st.info("Set TRADIER_TOKEN in Secrets.")
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
            # Main Charts
            net_val = processed[mode.lower()].sum() / 1e9
            st.metric(f"Total Net {mode}", f"{'-' if net_val < 0 else ''}${abs(net_val):,.2f}B")
            st.plotly_chart(render_plots(processed, ticker, S, mode), use_container_width=True)
            
            # --- WALL ANALYTICS ---
            st.markdown("---")
            st.subheader("🧱 Gamma Wall Analysis")
            
            agg_gex = processed.groupby('strike')['gex'].sum()
            call_wall = agg_gex.idxmax()
            put_wall = agg_gex.idxmin()
            flip_price = calculate_gamma_flip(processed)
            
            w1, w2, w3 = st.columns(3)
            w1.metric("Call Wall (Max +GEX)", f"${call_wall:,.0f}")
            w2.metric("Put Wall (Max -GEX)", f"${put_wall:,.0f}")
            w3.metric("Gamma Flip", f"${flip_price:,.2f}" if flip_price else "N/A")
            
            st.caption(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.warning("No data in range.")
    else:
        st.error("Connection failed.")

live_pulse()