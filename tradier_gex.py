import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
import pytz

# --- APP CONFIG ---
st.set_page_config(page_title="GEX & VANEX Pro", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    * { font-family: 'Arial', sans-serif !important; }
    .block-container { padding-top: 24px; padding-bottom: 8px; }
    [data-testid="stMetricValue"] { font-size: 22px !important; font-family: 'Arial' !important; }
    h1, h2, h3 { font-size: 18px !important; margin: 10px 0 6px 0 !important; font-weight: bold; }
    hr { margin: 15px 0 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SECRETS ---
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
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"API request failed: {e}")
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
                    if d.get('status') == 'open':
                        open_days.add(d.get('date'))
            except: pass
    return open_days

def fetch_data(ticker, max_exp):
    open_days = get_market_days()
    quote_data = tradier_get("markets/quotes", {"symbols": ticker})
    if not quote_data or 'quotes' not in quote_data: return None, None
    quote = quote_data['quotes']['quote']
    S = float(quote['last']) if isinstance(quote, dict) else float(quote[0]['last'])
    
    exp_data = tradier_get("markets/options/expirations", {"symbol": ticker, "includeAllRoots": "true"})
    if not exp_data or 'expirations' not in exp_data: return S, None
    all_exps = exp_data['expirations']['date']
    if not isinstance(all_exps, list): all_exps = [all_exps]
    valid_exps = [exp for exp in all_exps if exp in open_days][:max_exp]
    
    if not valid_exps: return S, None
    
    dfs = []
    prog = st.progress(0, text="Fetching chains...")
    for i, exp in enumerate(valid_exps):
        chain = tradier_get("markets/options/chains", {"symbol": ticker, "expiration": exp, "greeks": "true"})
        if chain and 'options' in chain and chain['options'] and chain['options']['option']:
            opts = chain['options']['option']
            df_chain = pd.DataFrame(opts) if isinstance(opts, list) else pd.DataFrame([opts])
            dfs.append(df_chain)
        prog.progress((i + 1) / len(valid_exps))
    prog.empty()
    return S, pd.concat(dfs, ignore_index=True) if dfs else None

def process_exposure(df, S, s_range):
    if df is None or df.empty: return pd.DataFrame()
    
    df["strike"] = pd.to_numeric(df["strike"], errors='coerce')
    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range)].copy()
    
    res = []
    for _, row in df.iterrows():
        g = row.get('greeks')
        if not g or not isinstance(g, dict): continue
        
        gamma = float(g.get('gamma') or 0)
        vega  = float(g.get('vega') or 0)
        iv    = float(g.get('smv_vol') or g.get('mid_iv') or 0)
        if iv > 1.0: iv /= 100.0
            
        oi = int(row.get('open_interest') or 0)
        op_type = row['option_type'].lower()
        side = 1 if op_type == 'call' else -1

        # Institutional Vanna Calculation
        # Vanna (dVega/dSpot) * Spot * 1% Vol Shift * 100 (Contract Size) * OI
        if S > 0 and iv > 0 and vega != 0:
            # This represents the $ change in Delta for a 1% move in IV
            vanna_raw = vega / (S * iv)
            vanex = side * vanna_raw * S * 0.01 * 100 * oi
        else:
            vanex = 0
            
        # GEX Calculation ($ Notional)
        # Gamma * Spot^2 * 0.01 * 100 * OI
        gex = side * gamma * (S**2) * 0.01 * 100 * oi
        
        res.append({
            "strike": row['strike'], 
            "expiry": row['expiration_date'],
            "gex": gex,
            "vanex": vanex,
            "type": op_type, 
            "oi": oi
        })
        
    return pd.DataFrame(res)

# -------------------------
# Heatmap Visualization
# -------------------------
def render_heatmap(df, ticker, S, mode):
    val_col = mode.lower()
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_col, aggfunc='sum').sort_index(ascending=False).fillna(0)
    z_raw, x_labs, y_labs = pivot.values, pivot.columns.tolist(), pivot.index.tolist()
    
    abs_limit = np.max(np.abs(z_raw)) if z_raw.size > 0 else 1.0
    closest_strike = min(y_labs, key=lambda x: abs(x - S))

    fig = go.Figure(data=go.Heatmap(
        z=z_raw, x=x_labs, y=y_labs,
        colorscale=CUSTOM_COLORSCALE, zmin=-abs_limit, zmax=abs_limit, zmid=0, zauto=False,
        colorbar=dict(title=f"{mode} ($)", tickfont=dict(family="Arial"))
    ))

    font_size = 11
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            label = f"${val/1e3:,.1f}K"
            if val < 0: t_color = "white"
            elif val > 0: t_color = "black"
            else: t_color = "yellow"

            fig.add_annotation(
                x=exp, y=strike, text=label, showarrow=False,
                font=dict(color=t_color, size=font_size, family="Arial Black")
            )

    calc_height = max(600, len(y_labs) * 30)
    fig.update_layout(
        title=f"{ticker} {mode} Matrix | Spot: ${S:,.2f}", 
        template="plotly_dark", height=calc_height, font=dict(family="Arial"),
        xaxis=dict(type='category', side='top'),
        yaxis=dict(title="Strike", tickmode='array', tickvals=y_labs,
            ticktext=[f"➔ <b>{s:,.0f}</b>" if s == closest_strike else f"{s:,.0f}" for s in y_labs])
    )
    return fig

# -------------------------
# Main Logic
# -------------------------
@st.fragment(run_every="60s")
def dashboard_content(ticker, max_exp, s_range):
    tz_est = pytz.timezone('US/Eastern')
    now_est = datetime.now(pytz.utc).astimezone(tz_est)
    st.write(f"🕒 Last updated: **{now_est.strftime('%H:%M:%S')} EST**")
    
    with st.spinner("Fetching market data..."):
        S, raw_df = fetch_data(ticker, int(max_exp))
        
    if S and raw_df is not None:
        df = process_exposure(raw_df, S, s_range)
        if not df.empty:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Net GEX", f"${df['gex'].sum()/1e9:,.2f}B")
            m2.metric("Net VANEX", f"${df['vanex'].sum()/1e6:,.1f}M")
            m3.metric("Total OI", f"{df['oi'].sum():,.0f}")
            m4.metric("Spot Price", f"${S:,.2f}")

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(render_heatmap(df, ticker, S, "GEX"), use_container_width=True)
            with c2:
                st.plotly_chart(render_heatmap(df, ticker, S, "VANEX"), use_container_width=True)

            # --- DIAGNOSTICS TABLE ---
            with st.expander("🔍 Data Diagnostics (Raw Greeks per Strike)"):
                st.write("If VANEX is 0, check if 'vega_raw' or 'iv_raw' are zero below.")
                diag_df = df[['strike', 'expiry', 'type', 'oi', 'vega_raw', 'iv_raw', 'vanex']].copy()
                st.dataframe(diag_df.sort_values(by='strike', ascending=False), use_container_width=True)
        else:
            st.warning("No data found. Try increasing the 'Strike ±' range.")
    else:
        st.error("API error. Verify your Tradier Token.")

def main():
    st.markdown("<h2 style='text-align:center;'>📊 GEX / VANEX Pro Analytics</h2>", unsafe_allow_html=True)
    cols = st.columns([1.5, 1, 1])
    ticker = cols[0].text_input("Ticker", value="SPY").upper().strip()
    max_exp = cols[1].number_input("Expiries", 1, 15, 5)
    s_range = cols[2].number_input("Strike ±", 5, 500, 25)
    dashboard_content(ticker, max_exp, s_range)

if __name__ == "__main__":
    main()