import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

# --- APP CONFIG ---
st.set_page_config(page_title="Tradier GEX Pro", page_icon="🏦", layout="wide")

# Tradier API Base URL
BASE_URL = "https://api.tradier.com/v1/"

# -------------------------
# Tradier Data Fetchers
# -------------------------
def get_tradier_data(endpoint, params, token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Tradier API Error: {response.status_code} - {response.text}")
        return None

def fetch_market_data(ticker, token):
    """Fetches current spot price."""
    data = get_tradier_data("markets/quotes", {"symbols": ticker}, token)
    if data and 'quotes' in data and 'quote' in data['quotes']:
        return data['quotes']['quote']['last']
    return None

def fetch_options_chain(ticker, expiry, token):
    """Fetches the full chain for a specific expiry."""
    params = {"symbol": ticker, "expiration": expiry, "greeks": "true"}
    data = get_tradier_data("markets/options/chains", params, token)
    if data and 'options' in data and data['options']:
        return pd.DataFrame(data['options']['option'])
    return pd.DataFrame()

# -------------------------
# Processing Logic
# -------------------------
def process_tradier_exposure(df, S, s_range, model_type):
    if df.empty: return pd.DataFrame()
    
    # Filter by Strike Range
    df = df[(df['strike'] >= S - s_range) & (df['strike'] <= S + s_range)].copy()
    
    # Tradier returns greeks directly. If missing, we skip or fallback.
    # Note: 'gex' calculation: Gamma * Spot^2 * 0.01 * 100 * OpenInterest
    res = []
    for _, row in df.iterrows():
        # Handle cases where Greeks might be None from API
        gamma = row.get('greeks', {}).get('gamma', 0) if row.get('greeks') else 0
        delta = row.get('greeks', {}).get('delta', 0) if row.get('greeks') else 0
        oi = row.get('open_interest', 0)
        
        if gamma is None: gamma = 0
        if delta is None: delta = 0
        
        # Exposure Math
        if model_type == "Dealer Short All (Absolute Stress)":
            gex = -gamma * S**2 * 0.01 * 100 * oi
        else:
            # Short Calls (Negative Gamma), Long Puts (Positive Gamma)
            gex = (-gamma if row['option_type'] == 'call' else gamma) * S**2 * 0.01 * 100 * oi

        dex = -delta * S * 100 * oi
        
        res.append({
            "strike": row['strike'],
            "expiry": row['expiration'],
            "gex": gex,
            "dex": dex
        })
        
    return pd.DataFrame(res)

# -------------------------
# Visualizations (Consistent with your style)
# -------------------------
def render_tradier_plots(df, ticker, S, mode):
    if df.empty: return None, None
    
    val_col = mode.lower()
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_col, aggfunc='sum').sort_index(ascending=False)
    
    z_raw = pivot.values
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** 0.5)
    y_labs = pivot.index.tolist()
    x_labs = pivot.columns.tolist()
    closest_strike = min(y_labs, key=lambda x: abs(x - S))

    # Heatmap
    fig_h = go.Figure(data=go.Heatmap(
        z=z_scaled, x=x_labs, y=y_labs,
        colorscale='Viridis', zmid=0, showscale=True,
        text=z_raw, hoverinfo="text" # Simplification for demo
    ))

    # Spot Highlight
    strike_diff = np.mean(np.diff(sorted(y_labs))) if len(y_labs) > 1 else 5
    fig_h.add_shape(
        type="rect", xref="paper", yref="y",
        x0=-0.08, x1=1.0, y0=closest_strike - (strike_diff*0.4), y1=closest_strike + (strike_diff*0.4),
        fillcolor="rgba(255, 51, 51, 0.25)", line=dict(width=0), layer="below"
    )

    fig_h.update_layout(
        title=f"Tradier {ticker} {mode} | Spot: ${S:,.2f}",
        template="plotly_dark", height=800,
        yaxis=dict(tickmode='array', tickvals=y_labs, 
                   ticktext=[f"<b>{s:,.0f}</b>" if s == closest_strike else f"{s:,.0f}" for s in y_labs])
    )

    return fig_h

# -------------------------
# Main Execution
# -------------------------
def main():
    st.sidebar.header("Tradier Auth")
    api_token = st.sidebar.text_input("Enter Bearer Token", type="password")
    
    ticker = st.sidebar.text_input("Symbol", "SPY").upper()
    mode = st.sidebar.radio("Metric", ["GEX", "DEX"])
    model_type = st.sidebar.selectbox("Model", ["Dealer Short All", "Standard"])
    s_range = st.sidebar.slider("Range", 5, 100, 30)

    if st.sidebar.button("Fetch Data") and api_token:
        with st.spinner("Accessing Tradier..."):
            S = fetch_market_data(ticker, api_token)
            
            # Fetch Expirations
            exp_data = get_tradier_data(f"markets/options/expirations", {"symbol": ticker}, api_token)
            if exp_data and S:
                expiries = exp_data['expirations']['date'][:5] # Get first 5 (includes 0DTE)
                
                all_chains = []
                for exp in expiries:
                    chain = fetch_options_chain(ticker, exp, api_token)
                    all_chains.append(chain)
                
                full_df = pd.concat(all_chains)
                processed = process_tradier_exposure(full_df, S, s_range, model_type)
                
                if not processed.empty:
                    # Metrics
                    t_val = processed[mode.lower()].sum() / 1e9
                    st.metric(f"Total {mode}", f"{'-' if t_val < 0 else ''}${abs(t_val):,.2f}B")
                    
                    fig = render_tradier_plots(processed, ticker, S, mode)
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()