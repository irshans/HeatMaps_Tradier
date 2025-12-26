import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# --- APP CONFIG ---
st.set_page_config(page_title="GEX Pro 2025", page_icon="📊", layout="wide")

if "TRADIER_TOKEN" in st.secrets:
    TRADIER_TOKEN = st.secrets["TRADIER_TOKEN"]
else:
    st.error("Please set TRADIER_TOKEN in Streamlit Secrets.")
    st.stop()

BASE_URL = "https://api.tradier.com/v1/"
CONTRACT_SIZE = 100

# --- YOUR CUSTOM COLOR SCHEME ---
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
    cal = tradier_get("markets/calendar", {})
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
    valid_exps = [d for d in all_exps if d in open_days][:max_exp]
    
    dfs = []
    prog = st.progress(0, text="Fetching Live Greeks...")
    for i, exp in enumerate(valid_exps):
        chain = tradier_get("markets/options/chains", {"symbol": ticker, "expiration": exp, "greeks": "true"})
        if chain and 'options' in chain and chain['options']:
            opts = chain['options']['option']
            dfs.append(pd.DataFrame(opts) if isinstance(opts, list) else pd.DataFrame([opts]))
        prog.progress((i+1)/len(valid_exps))
    prog.empty()
    return S, pd.concat(dfs) if dfs else None

# -------------------------
# Pure GEX Calculation
# -------------------------
def process_gex(df, S, s_range):
    if df is None or df.empty: return pd.DataFrame()
    df["strike"] = pd.to_numeric(df["strike"])
    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range)].copy()
    
    res = []
    for _, row in df.iterrows():
        g = row.get('greeks')
        if not g: continue
        gamma = float(g.get('gamma', 0) or 0)
        oi = int(row.get('open_interest', 0) or 0)
        
        # side: Call = +1 (Dealer Long), Put = -1 (Dealer Short)
        side = 1 if row['option_type'].lower() == 'call' else -1
        
        # Raw GEX Formula
        gex_val = side * gamma * (S**2) * 0.01 * CONTRACT_SIZE * oi
        
        res.append({
            "strike": row['strike'], "expiry": row['expiration_date'],
            "gex": gex_val
        })
    return pd.DataFrame(res)

def render_heatmap(df, ticker, S):
    pivot = df.pivot_table(index='strike', columns='expiry', values='gex', aggfunc='sum').sort_index(ascending=False).fillna(0)
    
    z = pivot.values
    x = [str(col) for col in pivot.columns]
    y = pivot.index.tolist()
    
    # Force symmetry for the colorscale center
    abs_max = np.max(np.abs(z)) if z.size else 1
    
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        colorscale=CUSTOM_COLORSCALE,
        zmid=0, zmin=-abs_max, zmax=abs_max,
        colorbar=dict(title="GEX Notional ($)")
    ))

    # Annotations - Simple and Clean
    for i, strike in enumerate(y):
        for j, exp in enumerate(x):
            val = z[i, j]
            if abs(val) < 500: continue
            
            # Find the Star
            star = " ★" if abs(val) == abs_max else ""
            
            # Contrast: Black for yellow/teal, White for purple
            f_color = "black" if val >= 0 else "white"
            
            fig.add_annotation(
                x=exp, y=strike, 
                text=f"${val/1e3:,.0f}K{star}", # Direct formatting
                showarrow=False,
                font=dict(color=f_color, size=10, family="Arial")
            )

    fig.update_layout(
        title=f"{ticker} GEX Heatmap | Spot: ${S:,.2f}", height=850, template="plotly_dark",
        font=dict(family="Arial"),
        xaxis=dict(type='category', title="Expiration Date"),
        yaxis=dict(title="Strike Price")
    )
    return fig

# -------------------------
# Main App
# -------------------------
def main():
    st.markdown("<h2 style='text-align:center; font-family:Arial;'>📊 GEX Pro (Native Greeks)</h2>", unsafe_allow_html=True)
    
    with st.sidebar:
        ticker = st.text_input("Symbol", "SPX").upper().strip()
        max_exp = st.slider("Expirations", 1, 10, 5)
        s_range = st.slider("Strike Range", 10, 300, 80)
        run = st.button("Calculate", type="primary")

    if run:
        S, raw_df = fetch_data(ticker, max_exp)
        if S and raw_df is not None:
            processed = process_gex(raw_df, S, s_range)
            if not processed.empty:
                st.plotly_chart(render_heatmap(processed, ticker, S), use_container_width=True)
            else: st.warning("Increase range or check ticker.")
        else: st.error("Fetch failed.")

if __name__ == "__main__":
    main()