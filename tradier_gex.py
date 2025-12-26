import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

# --- APP CONFIG ---
st.set_page_config(page_title="GEX Pro 2025", page_icon="📊", layout="wide")

# Force Arial globally via CSS
st.markdown("""
    <style>
    * { font-family: 'Arial', sans-serif !important; }
    .block-container { padding-top: 24px; padding-bottom: 8px; }
    button[kind="primary"], .stButton>button { padding:4px 8px !important; font-size:12px !important; height:30px !important; }
    input[type="text"], input[type="number"], select { padding:6px 8px !important; font-size:12px !important; height:28px !important; }
    [data-testid="stMetricValue"] { font-size: 22px !important; font-family: 'Arial' !important; }
    h1, h2, h3 { font-size: 18px !important; margin: 10px 0 6px 0 !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

if "TRADIER_TOKEN" in st.secrets:
    TRADIER_TOKEN = st.secrets["TRADIER_TOKEN"]
else:
    st.error("Please set TRADIER_TOKEN in Streamlit Secrets.")
    st.stop()

BASE_URL = "https://api.tradier.com/v1/"
CONTRACT_SIZE = 100

CUSTOM_COLORSCALE = [
    [0.00, '#050018'], [0.10, '#260446'], [0.25, '#56117a'],
    [0.40, '#6E298A'], [0.49, '#783F8F'], [0.50, '#224B8B'],
    [0.52, '#32A7A7'], [0.65, '#39B481'], [0.80, '#A8D42A'],
    [0.92, '#FFDF4A'], [1.00, '#F1F50C']
]

# -------------------------
# API & Processing
# -------------------------
def tradier_get(endpoint, params):
    headers = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers)
        if response.status_code == 200: return response.json()
    except: pass
    return None

@st.cache_data(ttl=3600)
def get_market_days():
    open_days = set()
    current_year = datetime.now().year
    for year in [current_year, current_year + 1]:
        for month in range(1, 13):
            cal = tradier_get("markets/calendar", {"month": month, "year": year})
            try:
                days = cal['calendar']['days']['day']
                if isinstance(days, dict): days = [days]
                for d in days:
                    if d.get('status') == 'open': open_days.add(d.get('date'))
            except: pass
    return open_days

def fetch_data(ticker, max_exp):
    open_days = get_market_days()
    quote_data = tradier_get("markets/quotes", {"symbols": ticker})
    if not quote_data: return None, None
    quote = quote_data['quotes']['quote']
    S = float(quote['last']) if isinstance(quote, dict) else float(quote[0]['last'])

    exp_data = tradier_get("markets/options/expirations", {"symbol": ticker, "includeAllRoots": "true"})
    if not exp_data: return S, None
    all_exps = exp_data['expirations']['date']
    if not isinstance(all_exps, list): all_exps = [all_exps]
    valid_exps = [exp for exp in all_exps if exp in open_days][:max_exp]
    
    dfs = []
    prog = st.progress(0, text="Reading Option Chains...")
    for i, exp in enumerate(valid_exps):
        chain = tradier_get("markets/options/chains", {"symbol": ticker, "expiration": exp, "greeks": "true"})
        if chain and 'options' in chain and chain['options']:
            opts = chain['options']['option']
            dfs.append(pd.DataFrame(opts) if isinstance(opts, list) else pd.DataFrame([opts]))
        prog.progress((i+1)/len(valid_exps))
    prog.empty()
    return S, pd.concat(dfs) if dfs else None

def process_exposure(df, S, s_range):
    if df is None or df.empty: return pd.DataFrame()
    df["strike"] = pd.to_numeric(df["strike"])
    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range)].copy()
    
    res = []
    for _, row in df.iterrows():
        g = row.get('greeks')
        if not g: continue
        gamma, vega = float(g.get('gamma', 0) or 0), float(g.get('vega', 0) or 0)
        oi, op_type = int(row.get('open_interest', 0) or 0), row['option_type'].lower()
        side = 1 if op_type == 'call' else -1
        
        res.append({
            "strike": row['strike'], "expiry": row['expiration_date'],
            "gex": side * gamma * (S**2) * 0.01 * CONTRACT_SIZE * oi,
            "vex": side * vega * 0.01 * CONTRACT_SIZE * oi,
            "type": op_type, "oi": oi
        })
    return pd.DataFrame(res)

# -------------------------
# Visualization
# -------------------------
def render_heatmap(df, ticker, S):
    pivot = df.pivot_table(index='strike', columns='expiry', values='gex', aggfunc='sum').sort_index(ascending=False).fillna(0)
    z_raw = pivot.values
    x_labs, y_labs = pivot.columns.tolist(), pivot.index.tolist()
    abs_limit = np.max(np.abs(z_raw)) if z_raw.size else 1.0
    
    # Identify the specific strike row for Spot
    closest_strike = min(y_labs, key=lambda x: abs(x - S))

    fig = go.Figure(data=go.Heatmap(
        z=z_raw, x=x_labs, y=y_labs, 
        colorscale=CUSTOM_COLORSCALE, zmin=-abs_limit, zmax=abs_limit, zmid=0,
        colorbar=dict(title="GEX ($)", tickfont=dict(family="Arial"))
    ))

    # Cell Annotations
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < 500: continue
            star = " ★" if abs(val) == abs_limit and abs_limit > 0 else ""
            label = f"${val/1e3:,.0f}K{star}"
            t_color = "black" if val >= 0 else "white"
            
            # If this is the spot strike, we can make the font bold or a different color
            is_spot_row = (strike == closest_strike)
            
            fig.add_annotation(
                x=exp, y=strike, text=label, showarrow=False,
                font=dict(
                    color="cyan" if is_spot_row and val < 0 else t_color, 
                    size=12, 
                    family="Arial",
                    weight="bold" if is_spot_row else "normal"
                )
            )

    # REFINED SPOT MARKER: Single dashed line across the specific strike
    fig.add_shape(
        type="line",
        xref="paper", yref="y",
        x0=0, x1=1,
        y0=closest_strike, y1=closest_strike,
        line=dict(color="rgba(255, 255, 255, 0.8)", width=3, dash="dot"),
    )

    calc_height = max(600, len(y_labs) * 25)

    fig.update_layout(
        title=f"{ticker} GEX Matrix | Spot: ${S:,.2f}", 
        template="plotly_dark", height=calc_height, font=dict(family="Arial"),
        xaxis=dict(type='category', side='top', tickfont=dict(size=12)),
        yaxis=dict(
            title="Strike", tickmode='array', tickvals=y_labs,
            # Visual indicator on the Y-Axis itself
            ticktext=[f"➔ <b>{s:,.0f}</b>" if s == closest_strike else f"{s:,.0f}" for s in y_labs],
            tickfont=dict(size=12)
        )
    )
    return fig

# -------------------------
# Main Execution
# -------------------------
def main():
    st.markdown("<h2 style='text-align:center;'>📊 GEX Pro Analytics</h2>", unsafe_allow_html=True)
    
    # Dynamic Strike Logic
    if "ticker_input" not in st.session_state:
        st.session_state.ticker_input = "SPX"
    
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 0.8])
    ticker = c1.text_input("Ticker", value=st.session_state.ticker_input).upper().strip()
    
    # Update default range based on ticker
    default_range = 80 if ticker == "SPX" else 25
    
    max_exp = c2.number_input("Expiries", 1, 15, 6)
    s_range = c3.number_input("Strike ±", 5, 500, default_range)
    run = c4.button("Run Sync", type="primary")

    if run:
        S, raw_df = fetch_data(ticker, int(max_exp))
        if S and raw_df is not None:
            df = process_exposure(raw_df, S, s_range)
            if not df.empty:
                net_gex = df["gex"].sum() / 1e9
                net_vex = df["vex"].sum() / 1e6
                calls = df[df["type"] == "call"]["oi"].sum()
                puts = df[df["type"] == "put"]["oi"].sum()
                cp_ratio = calls / puts if puts > 0 else 0
                
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Net GEX", f"${net_gex:,.2f}B")
                m2.metric("Net VEX", f"${net_vex:,.1f}M")
                m3.metric("Total Calls", f"{calls:,.0f}")
                m4.metric("Total Puts", f"{puts:,.0f}")
                m5.metric("Call/Put Ratio", f"{cp_ratio:.2f}")
                
                st.plotly_chart(render_heatmap(df, ticker, S), use_container_width=True)
            else: st.warning("No data found in range.")
        else: st.error("Fetch failed.")

if __name__ == "__main__":
    main()