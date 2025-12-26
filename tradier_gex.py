import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
from datetime import datetime

# --- CREDENTIALS LOADING ---
# Checks Streamlit Cloud Secrets first, then local environment
if "TRADIER_TOKEN" in st.secrets:
    saved_token = st.secrets["TRADIER_TOKEN"]
else:
    # Fallback for local testing (requires a .env file)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        saved_token = os.getenv("TRADIER_TOKEN", "")
    except ImportError:
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
    except Exception:
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
        # Convert to list if single object returned
        return pd.DataFrame(opts) if isinstance(opts, list) else pd.DataFrame([opts])
    return pd.DataFrame()

# -------------------------
# Processing Logic
# -------------------------
def process_exposure(df, S, s_range, model_type):
    if df.empty: return pd.DataFrame()
    
    # Filter by strike proximity
    df = df[(df['strike'] >= S - s_range) & (df['strike'] <= S + s_range)].copy()
    
    res = []
    for _, row in df.iterrows():
        greeks = row.get('greeks', {})
        if not greeks or greeks is None: continue
        
        gamma = greeks.get('gamma', 0) or 0
        delta = greeks.get('delta', 0) or 0
        oi = row.get('open_interest', 0) or 0
        
        # Exposure Calculations
        if model_type == "Dealer Short All (Absolute Stress)":
            gex = -gamma * S**2 * 0.01 * CONTRACT_SIZE * oi
        else:
            # Standard Model: Short Calls (-), Long Puts (+)
            gex = (-gamma if row['option_type'] == 'call' else gamma) * S**2 * 0.01 * CONTRACT_SIZE * oi

        dex = -delta * S * CONTRACT_SIZE * oi
        
        res.append({
            "strike": row['strike'],
            "expiry": row['expiration_date'], # FIXED: Match Tradier API key
            "gex": gex,
            "dex": dex
        })
        
    return pd.DataFrame(res)

# -------------------------
# Plotting
# -------------------------
def render_plots(df, ticker, S, mode):
    val_col = mode.lower()
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_col, aggfunc='sum').sort_index(ascending=False)
    
    z_raw = pivot.values
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** 0.5)
    y_labs = pivot.index.tolist()
    x_labs = pivot.columns.tolist()
    closest_strike = min(y_labs, key=lambda x: abs(x - S))

    # Tooltip Formatting
    h_text = []
    for i, strike in enumerate(y_labs):
        row = []
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            prefix = "-" if val < 0 else ""
            v_abs = abs(val)
            formatted = f"{prefix}${v_abs/1e6:,.2f}M" if v_abs >= 1e6 else f"{prefix}${v_abs/1e3:,.1f}K"
            row.append(f"Strike: ${strike:,.0f}<br>Exp: {exp}<br>{mode}: {formatted}")
        h_text.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_scaled, x=x_labs, y=y_labs, text=h_text, hoverinfo="text",
        colorscale='Viridis', zmid=0, showscale=True
    ))

    # Annotations & Red Spot Highlight
    strike_diff = np.mean(np.diff(sorted(y_labs))) if len(y_labs) > 1 else 5
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < 500: continue
            txt = f"{'-' if val < 0 else ''}${abs(val)/1e3:,.0f}K"
            fig.add_annotation(x=exp, y=strike, text=txt, showarrow=False, font=dict(color="white", size=11))

    fig.add_shape(type="rect", xref="paper", yref="y", x0=-0.05, x1=1.05, 
                  y0=closest_strike-(strike_diff*0.4), y1=closest_strike+(strike_diff*0.4),
                  fillcolor="rgba(255, 0, 0, 0.25)", line=dict(width=0), layer="below")

    fig.update_layout(
        title=f"LIVE {ticker} {mode} EXPOSURE | Spot: ${S:,.2f}",
        template="plotly_dark", height=850,
        xaxis=dict(type='category', side='top'),
        yaxis=dict(tickmode='array', tickvals=y_labs, 
                   ticktext=[f"<b>{s:,.0f}</b>" if s == closest_strike else f"{s:,.0f}" for s in y_labs])
    )
    return fig

# -------------------------
# Live Refresh Component
# -------------------------
@st.fragment(run_every="60s")
def live_dashboard(ticker, api_token, max_exp, s_range, model_type, mode):
    with st.spinner("Refreshing pulse..."):
        S = fetch_market_data(ticker, api_token)
        exp_data = get_tradier_data("markets/options/expirations", {"symbol": ticker}, api_token)
        
        if S and exp_data:
            dates = exp_data['expirations']['date']
            if not isinstance(dates, list): dates = [dates]
            
            # Efficiently fetch and process chains
            all_chains = [fetch_options_chain(ticker, d, api_token) for d in dates[:max_exp]]
            full_df = pd.concat(all_chains)
            processed = process_exposure(full_df, S, s_range, model_type)
            
            if not processed.empty:
                val = processed[mode.lower()].sum() / 1e9
                st.metric(f"Total Net {mode} (Updating Live)", f"{'-' if val < 0 else ''}${abs(val):,.2f}B")
                st.plotly_chart(render_plots(processed, ticker, S, mode), use_container_width=True)
                st.caption(f"Last API Sync: {datetime.now().strftime('%H:%M:%S')}")
            else:
                st.warning("No data found in range. Check ticker or strike range.")
        else:
            st.error("API failed to return data. Verify token and ticker.")

# -------------------------
# Main App
# -------------------------
def main():
    st.sidebar.title("⚡ GEX Live Pulse")
    
    # Sidebar inputs
    api_token = st.sidebar.text_input("Tradier Token", value=saved_token, type="password")
    ticker = st.sidebar.text_input("Ticker Symbol", "SPY").upper().strip()
    mode = st.sidebar.radio("Analysis Mode", ["GEX", "DEX"])
    model_type = st.sidebar.selectbox("Dealer Model", ["Standard (Call/Put Mix)", "Dealer Short All (Absolute Stress)"])
    max_exp = st.sidebar.slider("Expirations to Load", 1, 10, 5)
    s_range = st.sidebar.slider("Strike Range (±)", 5, 200, 30)

    if api_token:
        live_dashboard(ticker, api_token, max_exp, s_range, model_type, mode)
    else:
        st.warning("Please enter your Tradier Token or set it in Streamlit Secrets.")

if __name__ == "__main__":
    main()