import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

# --- APP CONFIG ---
st.set_page_config(page_title="GEX Pro 2025", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 24px; padding-bottom: 8px; }
    button[kind="primary"], .stButton>button { padding:4px 8px !important; font-size:12px !important; height:30px !important; }
    input[type="text"], input[type="number"], select { padding:6px 8px !important; font-size:12px !important; height:28px !important; }
    h1, h2, h3 { font-size: 18px !important; margin: 10px 0 6px 0 !important; }
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
# Tradier API Methods
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
    prog = st.progress(0, text="Fetching Greeks...")
    for i, exp in enumerate(valid_exps):
        chain = tradier_get("markets/options/chains", {"symbol": ticker, "expiration": exp, "greeks": "true"})
        if chain and 'options' in chain and chain['options']:
            opts = chain['options']['option']
            dfs.append(pd.DataFrame(opts) if isinstance(opts, list) else pd.DataFrame([opts]))
        prog.progress((i+1)/len(valid_exps))
    prog.empty()
    return S, pd.concat(dfs) if dfs else None

# -------------------------
# GEX Calculation
# -------------------------
def process_gex(df, S, s_range):
    if df is None or df.empty: return pd.DataFrame()
    df["strike"] = pd.to_numeric(df["strike"])
    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range)].copy()
    
    res = []
    for _, row in df.iterrows():
        g = row.get('greeks')
        if not g: continue
        gamma, vega = float(g.get('gamma', 0) or 0), float(g.get('vega', 0) or 0)
        oi, op_type = int(row.get('open_interest', 0) or 0), row['option_type'].lower()
        
        # Dealer: Short Calls (-1) / Long Puts (+1)
        dealer_pos = -1 if op_type == 'call' else 1
        
        res.append({
            "strike": row['strike'], "expiry": row['expiration_date'],
            "gex": dealer_pos * gamma * (S**2) * 0.01 * CONTRACT_SIZE * oi,
            "vex": dealer_pos * vega * 0.01 * CONTRACT_SIZE * oi,
            "option_type": op_type, "oi": oi
        })
    return pd.DataFrame(res)

# -------------------------
# Visualization Functions
# -------------------------
def render_gamma_wall_chart(df, S):
    if df.empty: return None
    agg = df.groupby('strike')['gex'].sum().sort_index()
    colors = ['#2563eb' if v < 0 else '#fbbf24' for v in agg.values]
    
    fig = go.Figure(go.Bar(x=agg.index, y=agg.values, marker_color=colors))
    fig.add_vline(x=S, line_color="white", line_dash="dot", annotation_text="Spot")
    fig.update_layout(title="Net GEX by Strike", template="plotly_dark", height=400)
    return fig

def render_heatmap(df, ticker, S):
    pivot = df.pivot_table(index='strike', columns='expiry', values='gex', aggfunc='sum').sort_index(ascending=False).fillna(0)
    z_raw = pivot.values
    x_labs, y_labs = pivot.columns.tolist(), pivot.index.tolist()
    
    abs_max = np.max(np.abs(z_raw)) if z_raw.size else 1.0
    
    fig = go.Figure(data=go.Heatmap(
        z=z_raw, x=x_labs, y=y_labs, 
        colorscale=CUSTOM_COLORSCALE, zmin=-abs_max, zmax=abs_max, zmid=0,
        colorbar=dict(title="GEX ($)")
    ))

    # Cell Annotations
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < 500: continue
            
            # Label Building (Negative sign included automatically)
            label = f"${val/1e3:,.0f}K"
            if abs(val) == abs_max: label += " ⭐"
            
            # Contrast logic: Yellow-side (pos) = Black text, Purple-side (neg) = White text
            t_color = "black" if val > 0 else "white"
            
            fig.add_annotation(x=exp, y=strike, text=label, showarrow=False,
                               font=dict(color=t_color, size=11, family="Arial"))

    fig.update_layout(
        title=f"{ticker} GEX Heatmap | Spot: ${S:,.2f}", 
        template="plotly_dark", height=850,
        xaxis=dict(type='category', side='top'),
        yaxis=dict(title="Strike")
    )
    return fig

# -------------------------
# Main App
# -------------------------
def main():
    st.markdown("<div style='text-align:center;'><h2 style='font-size:18px;'>📊 GEX Pro (Tradier)</h2></div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 0.8])
    with col1: ticker = st.text_input("Ticker", "SPX").upper().strip()
    with col2: max_exp = st.number_input("Max Exp", 1, 15, 5)
    with col3: s_range = st.number_input("Strike ±", 10, 500, 80)
    with col4: run = st.button("Run", type="primary")

    if run:
        S, raw_df = fetch_data(ticker, int(max_exp))
        if S and raw_df is not None:
            processed = process_gex(raw_df, S, s_range)
            if not processed.empty:
                # Render UI
                st.plotly_chart(render_heatmap(processed, ticker, S), use_container_width=True)
                st.plotly_chart(render_gamma_wall_chart(processed, S), use_container_width=True)
            else: st.warning("No data found.")
        else: st.error("Fetch failed.")

if __name__ == "__main__":
    main()