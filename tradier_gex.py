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

def tradier_get(endpoint, params):
    headers = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers)
        if response.status_code == 200: return response.json()
    except: pass
    return None

@st.cache_data(ttl=3600)
def get_open_market_days():
    open_days = set()
    cal_data = tradier_get("markets/calendar", {})
    try:
        if cal_data and 'calendar' in cal_data:
            days = cal_data['calendar']['days']['day']
            if isinstance(days, dict): days = [days]
            for d in days:
                if d.get('status') == 'open':
                    open_days.add(d.get('date'))
    except: pass
    return open_days

def fetch_tradier_data(ticker, max_exp):
    open_days = get_open_market_days()
    quote_data = tradier_get("markets/quotes", {"symbols": ticker})
    if not quote_data or 'quotes' not in quote_data: return None, None
    quote = quote_data['quotes']['quote']
    S = float(quote['last']) if isinstance(quote, dict) else float(quote[0]['last'])

    exp_data = tradier_get("markets/options/expirations", {"symbol": ticker, "includeAllRoots": "true"})
    if not exp_data or 'expirations' not in exp_data: return S, None
    
    all_exps = exp_data['expirations']['date']
    if not isinstance(all_exps, list): all_exps = [all_exps]
    valid_exps = [d for d in all_exps if d in open_days][:max_exp]
    
    dfs = []
    prog = st.progress(0, text="Syncing Option Chain...")
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
    # Ensure strike is numeric
    df["strike"] = pd.to_numeric(df["strike"])
    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range)].copy()
    
    res = []
    for _, row in df.iterrows():
        g = row.get('greeks')
        if not g or not isinstance(g, dict): continue
        
        gamma = float(g.get('gamma', 0) or 0)
        vanna = float(g.get('vanna', 0) or 0)
        oi = int(row.get('open_interest', 0) or 0)
        
        # --- THE FIX: FORCED POSITIONING ---
        # Call = Dealer Long (+), Put = Dealer Short (-)
        is_call = row['option_type'].lower() == 'call'
        pos = 1 if is_call else -1
        
        res.append({
            "strike": row['strike'], 
            "expiry": row['expiration_date'],
            "gex": pos * gamma * (S**2) * 0.01 * CONTRACT_SIZE * oi,
            "vex": pos * vanna * S * 0.01 * CONTRACT_SIZE * oi
        })
    return pd.DataFrame(res)

def render_plots(df, ticker, S, mode):
    val_col = mode.lower()
    
    # Aggregate for Bars
    agg = df.groupby('strike')[val_col].sum().sort_index()
    
    # Pivot for Heatmap
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_col, aggfunc='sum').sort_index(ascending=False)
    pivot = pivot.fillna(0)
    
    z_raw = pivot.values
    x_labs = [str(x) for x in pivot.columns.tolist()]
    y_labs = pivot.index.tolist()
    
    # Find absolute max for scaling and the star
    abs_limit = np.max(np.abs(z_raw)) if z_raw.size else 1
    
    fig_h = go.Figure(data=go.Heatmap(
        z=z_raw, x=x_labs, y=y_labs, 
        colorscale='RdBu', zmid=0, zmin=-abs_limit, zmax=abs_limit,
        colorbar=dict(title=f"{mode} Notional")
    ))

    # Add Labels & Star
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < 100: continue # Hide noise
            
            star = " ★" if abs(val) == abs_limit else ""
            # Manual sign display in text
            sign_str = "+" if val > 0 else "-" if val < 0 else ""
            label_text = f"{sign_str}${abs(val)/1e3:,.0f}K{star}"
            
            f_color = "white" if abs(val) > (abs_limit * 0.3) else "silver"
            if abs(val) < (abs_limit * 0.1): f_color = "black"

            fig_h.add_annotation(
                x=exp, y=strike, text=label_text,
                showarrow=False, font=dict(color=f_color, size=10, family="Arial")
            )

    fig_h.update_layout(
        title=f"{ticker} {mode} Matrix | Spot: ${S:,.2f}", template="plotly_dark", height=800,
        font=dict(family="Arial"),
        xaxis=dict(type='category', title="Expiration"),
        yaxis=dict(title="Strike")
    )

    # Bar chart
    colors = ['#ef4444' if v < 0 else '#10b981' for v in agg.values]
    fig_b = go.Figure(go.Bar(x=agg.index, y=agg.values, marker_color=colors))
    fig_b.update_layout(title=f"Net {mode} by Strike", template="plotly_dark", height=350, font=dict(family="Arial"))
    
    return fig_h, fig_b

def main():
    st.markdown("<h2 style='text-align:center; font-family:Arial;'>📊 GEX Pro (Market Maker View)</h2>", unsafe_allow_html=True)
    ticker = st.text_input("Symbol", "SPX").upper().strip()
    
    colA, colB, colC = st.columns(3)
    with colA: mode = st.radio("Metric", ["GEX", "VEX"], horizontal=True)
    with colB: max_exp = st.number_input("Lookahead (Exps)", 1, 10, 5)
    with colC: s_range = st.number_input("Strike Range", 10, 500, 75)
    
    if st.button("Calculate Exposure", type="primary"):
        S, raw_df = fetch_tradier_data(ticker, int(max_exp))
        if S and raw_df is not None:
            processed = process_exposure(raw_df, S, s_range)
            if not processed.empty:
                net = processed[mode.lower()].sum()
                st.metric(f"Total Net {mode}", f"${net/1e9:,.2f}B", delta=f"{net/1e6:,.0f}M")
                h, b = render_plots(processed, ticker, S, mode)
                st.plotly_chart(h, use_container_width=True)
                st.plotly_chart(b, use_container_width=True)
            else: st.warning("No data for current filters.")
        else: st.error("Failed to fetch data.")

if __name__ == "__main__":
    main()