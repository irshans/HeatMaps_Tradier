import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# --- APP CONFIG ---
st.set_page_config(page_title="GEX Pro 2025", page_icon="📊", layout="wide")

# Compact UI styling
st.markdown(
    """
    <style>
    /* Increase top padding so title is visible and not cut off */
    .block-container { padding-top: 24px; padding-bottom: 8px; }

    /* Buttons */
    button[kind="primary"], .stButton>button {
        padding:4px 8px !important;
        font-size:12px !important;
        height:30px !important;
    }

    /* Inputs, selects, number inputs */
    input[type="text"], input[type="number"], select {
        padding:6px 8px !important;
        font-size:12px !important;
        height:28px !important;
    }

    /* Radio, selectbox height & font */
    div[role="radiogroup"] label, .stSelectbox, .stRadio {
        font-size:12px !important;
    }

    /* Sliders compact */
    .stSlider > div, .stNumberInput > div {
        font-size:12px !important;
        height:34px !important;
    }

    /* Reduce margins for columns */
    .css-1lcbmhc.e1tzin5v0 { gap: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "TRADIER_TOKEN" in st.secrets:
    TRADIER_TOKEN = st.secrets["TRADIER_TOKEN"]
else:
    st.error("Please set TRADIER_TOKEN in Streamlit Secrets.")
    st.stop()

BASE_URL = "https://api.tradier.com/v1/"
CONTRACT_SIZE = 100

# Updated color scheme with narrow neutral band
CUSTOM_COLORSCALE = [
    [0.00, '#050018'],  # deepest purple (extreme negative)
    [0.10, '#260446'],
    [0.25, '#56117a'],
    [0.40, '#6E298A'],
    [0.49, '#783F8F'],  # last purple before center
    [0.50, '#224B8B'],  # explicit center (neutral blue)
    [0.52, '#32A7A7'],  # light teal (small positive)
    [0.65, '#39B481'],  # greenish
    [0.80, '#A8D42A'],  # yellow-green
    [0.92, '#FFDF4A'],
    [1.00, '#F1F50C']   # bright yellow (extreme positive)
]

# -------------------------
# Tradier API Methods
# -------------------------
def tradier_get(endpoint, params):
    headers = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers)
        if response.status_code == 200: 
            return response.json()
    except: 
        pass
    return None

@st.cache_data(ttl=3600)
def get_market_days():
    open_days = set()
    cal = tradier_get("markets/calendar", {})
    try:
        days = cal['calendar']['days']['day']
        if isinstance(days, dict): 
            days = [days]
        for d in days:
            if d.get('status') == 'open': 
                open_days.add(d.get('date'))
    except: 
        pass
    return open_days

def fetch_data(ticker, max_exp):
    open_days = get_market_days()
    quote_data = tradier_get("markets/quotes", {"symbols": ticker})
    if not quote_data: 
        return None, None
    quote = quote_data['quotes']['quote']
    S = float(quote['last']) if isinstance(quote, dict) else float(quote[0]['last'])

    exp_data = tradier_get("markets/options/expirations", {"symbol": ticker, "includeAllRoots": "true"})
    if not exp_data: 
        return S, None
    all_exps = exp_data['expirations']['date']
    if not isinstance(all_exps, list): 
        all_exps = [all_exps]
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
    if df is None or df.empty: 
        return pd.DataFrame()
    df["strike"] = pd.to_numeric(df["strike"])
    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range)].copy()
    
    res = []
    for _, row in df.iterrows():
        g = row.get('greeks')
        if not g: 
            continue
        gamma = float(g.get('gamma', 0) or 0)
        oi = int(row.get('open_interest', 0) or 0)
        
        # Dealer model: Short Calls (-1) / Long Puts (+1)
        dealer_pos = -1 if row['option_type'].lower() == 'call' else 1
        
        # GEX Formula
        gex_val = dealer_pos * gamma * (S**2) * 0.01 * CONTRACT_SIZE * oi
        
        res.append({
            "strike": row['strike'], 
            "expiry": row['expiration_date'],
            "gex": gex_val
        })
    return pd.DataFrame(res)

def render_heatmap(df, ticker, S):
    pivot = df.pivot_table(index='strike', columns='expiry', values='gex', aggfunc='sum').sort_index(ascending=False).fillna(0)
    
    z = pivot.values
    x = [str(col) for col in pivot.columns]
    y = pivot.index.tolist()
    
    if not y:
        return None
    
    # Symmetric range for proper color mapping
    abs_max = np.max(np.abs(z)) if z.size else 1
    
    # Find closest strike to spot
    closest_strike = min(y, key=lambda x: abs(x - S))
    
    # Build hover text
    h_text = []
    for i, strike in enumerate(y):
        row = []
        for j, exp in enumerate(x):
            val = z[i, j]
            prefix = "-" if val < 0 else ""
            v_abs = abs(val)
            if v_abs >= 1e6:
                formatted = f"{prefix}${v_abs/1e6:,.2f}M"
            elif v_abs >= 1e3:
                formatted = f"{prefix}${v_abs/1e3:,.1f}K"
            else:
                formatted = f"{prefix}${v_abs:,.0f}"
            row.append(f"Strike: ${strike:,.0f}<br>Expiry: {exp}<br>GEX: {formatted}")
        h_text.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y, text=h_text, hoverinfo="text",
        colorscale=CUSTOM_COLORSCALE,
        zmid=0, zmin=-abs_max, zmax=abs_max,
        colorbar=dict(title="GEX ($)", tickformat="$,.2s")
    ))

    # Annotations - show values above threshold
    max_abs_val = np.max(np.abs(z)) if z.size else 0
    for i, strike in enumerate(y):
        for j, exp in enumerate(x):
            val = z[i, j]
            if abs(val) < 500: 
                continue
            
            prefix = "-" if val < 0 else ""
            txt = f"{prefix}${abs(val)/1e3:,.0f}K"
            if abs(val) == max_abs_val and max_abs_val > 0:
                txt += " ⭐"
            
            # Text color based on value
            text_color = "black" if val >= 0 else "white"
            
            fig.add_annotation(
                x=exp, y=strike, 
                text=txt, 
                showarrow=False,
                font=dict(color=text_color, size=12, family="Arial")
            )

    # Highlight spot strike
    sorted_strikes = sorted(y)
    strike_diffs = np.diff(sorted_strikes) if len(sorted_strikes) > 1 else np.array([sorted_strikes[0] * 0.05])
    padding = (strike_diffs[0] * 0.45) if len(strike_diffs) > 0 else 2.5

    fig.add_shape(
        type="rect", xref="paper", yref="y",
        x0=-0.08, x1=1.0,
        y0=closest_strike - padding,
        y1=closest_strike + padding,
        fillcolor="rgba(255, 51, 51, 0.25)",
        line=dict(width=0),
        layer="below"
    )

    fig.update_layout(
        title=f"{ticker} GEX Heatmap | Spot: ${S:,.2f} — Dealer: Short Calls / Long Puts",
        height=900,
        template="plotly_dark",
        font=dict(family="Arial"),
        xaxis=dict(type='category', title="Expiration Date", side='top', tickfont=dict(size=12)),
        yaxis=dict(
            title="Strike Price",
            tickfont=dict(size=12),
            autorange=True,
            tickmode='array',
            tickvals=y,
            ticktext=[f"<b>{s:,.0f}</b>" if s == closest_strike else f"{s:,.0f}" for s in y]
        ),
        margin=dict(l=80, r=60, t=100, b=40)
    )
    return fig

# -------------------------
# Main App
# -------------------------
def main():
    # Small centered title
    st.markdown(
        "<div style='text-align:center; margin-top:6px;'><h2 style='font-size:18px; margin:10px 0 6px 0; font-weight:600;'>📊 GEX Pro (Tradier API)</h2></div>",
        unsafe_allow_html=True,
    )
    
    # Compact single-line toolbar
    col1, col2, col3, col4 = st.columns([1.8, 0.9, 0.9, 0.8])
    with col1:
        ticker = st.text_input("Ticker", "SPX", key="ticker_compact").upper().strip()
    with col2:
        max_exp = st.number_input("Max Exp", min_value=1, max_value=15, value=5, step=1, key="maxexp_compact")
    with col3:
        s_range = st.number_input("Strike ±", min_value=10, max_value=300, value=80, step=5, key="srange_compact")
    with col4:
        run = st.button("Run", type="primary", key="run_compact")

    if run:
        with st.spinner(f"Analyzing {ticker}..."):
            S, raw_df = fetch_data(ticker, int(max_exp))
        
        if S and raw_df is not None:
            processed = process_gex(raw_df, S, s_range)
            if not processed.empty:
                # Display net GEX metric
                t_gex = processed["gex"].sum() / 1e9
                p_g = "-" if t_gex < 0 else ""
                st.metric("Net Dealer GEX", f"{p_g}${abs(t_gex):,.2f}B")
                
                # Render heatmap
                fig = render_heatmap(processed, ticker, S)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else: 
                st.warning("No data in range. Broaden strike range or check ticker.")
        else: 
            st.error("Fetch failed. Check API token or market hours.")

if __name__ == "__main__":
    main()