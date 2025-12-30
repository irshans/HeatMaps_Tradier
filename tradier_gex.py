import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import pytz

# --- APP CONFIG ---
st.set_page_config(page_title="GEX & VANEX Institutional Pro", page_icon="🏦", layout="wide")

# Custom CSS for UI polish
st.markdown("""
    <style>
    * { font-family: 'Arial', sans-serif !important; }
    .stButton>button { border-radius: 5px; height: 3em; font-weight: bold; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- SECRETS ---
if "TRADIER_TOKEN" in st.secrets:
    TRADIER_TOKEN = st.secrets["TRADIER_TOKEN"]
else:
    st.error("Missing TRADIER_TOKEN in secrets.")
    st.stop()

BASE_URL = "https://api.tradier.com/v1/"

CUSTOM_COLORSCALE = [
    [0.00, '#050018'], [0.10, '#260446'], [0.25, '#56117a'],
    [0.40, '#6E298A'], [0.49, '#783F8F'], [0.50, '#224B8B'],
    [0.52, '#32A7A7'], [0.65, '#39B481'], [0.80, '#A8D42A'],
    [0.92, '#FFDF4A'], [1.00, '#F1F50C']
]

# -------------------------
# API Helper
# -------------------------
def tradier_get(endpoint, params):
    headers = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers, timeout=15)
        if response.status_code == 200: return response.json()
    except: pass
    return None

@st.cache_data(ttl=3600)
def get_market_days():
    open_days = set()
    dt = datetime.now()
    for month_offset in [0, 1]:
        cal = tradier_get("markets/calendar", {"month": dt.month, "year": dt.year})
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
    prog = st.progress(0, text="Fetching chains...")
    for i, exp in enumerate(valid_exps):
        chain = tradier_get("markets/options/chains", {"symbol": ticker, "expiration": exp, "greeks": "true"})
        if chain and 'options' in chain and chain['options']:
            opts = chain['options']['option']
            dfs.append(pd.DataFrame(opts) if isinstance(opts, list) else pd.DataFrame([opts]))
        prog.progress((i + 1) / len(valid_exps))
    prog.empty()
    return S, pd.concat(dfs, ignore_index=True) if dfs else None

# -------------------------
# Processing
# -------------------------
def process_data(df, S, s_range):
    if df is None or df.empty: return pd.DataFrame()
    df["strike"] = pd.to_numeric(df["strike"], errors='coerce')
    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range)].copy()
    
    res = []
    for _, row in df.iterrows():
        g = row.get('greeks')
        if not g: continue
        
        gamma = float(g.get('gamma') or 0)
        vega  = float(g.get('vega') or 0)
        iv = float(g.get('smv_vol') or g.get('mid_iv') or 0)
        if iv > 1.0: iv /= 100.0
            
        oi = int(row.get('open_interest') or 0)
        side = 1 if row['option_type'].lower() == 'call' else -1
        
        # Institutional Scaled Notional
        gex = side * gamma * (S**2) * 0.01 * 100 * oi
        vanex = (side * (vega / (S * iv)) * S * 0.01 * 100 * oi) if (S > 0 and iv > 0) else 0
            
        res.append({"strike": row['strike'], "expiry": row['expiration_date'], 
                    "gex": gex, "vanex": vanex, "oi": oi})
    return pd.DataFrame(res)

def render_heatmap(df, ticker, S, mode):
    pivot = df.pivot_table(index='strike', columns='expiry', values=mode.lower(), aggfunc='sum').sort_index(ascending=False).fillna(0)
    z = pivot.values
    abs_max = np.max(np.abs(z)) if z.size > 0 else 1
    
    fig = go.Figure(data=go.Heatmap(
        z=z, x=pivot.columns, y=pivot.index,
        colorscale=CUSTOM_COLORSCALE, zmin=-abs_max, zmax=abs_max, zmid=0
    ))

    for i, strike in enumerate(pivot.index):
        for j, exp in enumerate(pivot.columns):
            val = z[i, j]
            label = f"{val/1e6:.1f}M" if abs(val) >= 1e6 else (f"{val/1e3:.0f}K" if abs(val) >= 1e3 else f"{val:.0f}")
            fig.add_annotation(x=exp, y=strike, text=label, showarrow=False, font=dict(color="white" if val < 0 else "black", size=10))

    fig.update_layout(title=f"{ticker} {mode} Matrix", template="plotly_dark", height=700, xaxis=dict(side='top'))
    return fig

# -------------------------
# Main Page
# -------------------------
def main():
    st.title("📊 Institutional GEX / VANEX Dashboard")

    # Sidebar Controls
    ticker = st.sidebar.text_input("Ticker", "SPY").upper().strip()
    max_exp = st.sidebar.slider("Expiries", 1, 15, 5)
    s_range = st.sidebar.number_input("Strike Range ±", 5, 500, 25)

    # Manual Refresh Button (Main Area)
    if st.button("🔄 Force Data Refresh"):
        st.cache_data.clear()
        st.rerun()

    # 10-Minute Fragment
    @st.fragment(run_every="600s")
    def dashboard_fragment():
        now = datetime.now(pytz.timezone('US/Eastern')).strftime("%H:%M:%S")
        st.caption(f"Next Auto-Refresh in 10 minutes | Last Update: {now} EST")
        
        S, raw_df = fetch_data(ticker, max_exp)
        if raw_df is not None:
            df = process_data(raw_df, S, s_range)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Net GEX Notional", f"${df['gex'].sum()/1e6:.1f}M")
            m2.metric("Net VANEX Notional", f"${df['vanex'].sum()/1e6:.1f}M")
            m3.metric("Spot Price", f"${S:.2f}")
            
            t1, t2 = st.tabs(["Gamma Exposure", "Vanna Exposure"])
            with t1: st.plotly_chart(render_heatmap(df, ticker, S, "GEX"), width="stretch")
            with t2: st.plotly_chart(render_heatmap(df, ticker, S, "VANEX"), width="stretch")
        else:
            st.error("No data returned. Verify ticker and API Token.")

    dashboard_fragment()

if __name__ == "__main__":
    main()