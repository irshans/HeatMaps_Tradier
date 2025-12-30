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
    /* Styling the diagnostic table */
    [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- SECRETS ---
if "TRADIER_TOKEN" in st.secrets:
    TRADIER_TOKEN = st.secrets["TRADIER_TOKEN"]
else:
    st.error("Please set TRADIER_TOKEN in Streamlit Secrets.")
    st.stop()

BASE_URL = "https://api.tradier.com/v1/"

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
        if response.status_code == 200: return response.json()
    except: pass
    return None

@st.cache_data(ttl=3600)
def get_market_days():
    open_days = set()
    dt = datetime.now()
    for month_offset in [0, 1]:
        target_dt = dt if month_offset == 0 else (dt.replace(day=1) + pd.DateOffset(months=1))
        cal = tradier_get("markets/calendar", {"month": target_dt.month, "year": target_dt.year})
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
    
    valid_exps = sorted([exp for exp in all_exps if exp in open_days])[:max_exp]
    
    dfs = []
    prog = st.progress(0, text="Fetching data...")
    for i, exp in enumerate(valid_exps):
        chain = tradier_get("markets/options/chains", {"symbol": ticker, "expiration": exp, "greeks": "true"})
        if chain and 'options' in chain and chain['options'] and chain['options']['option']:
            opts = chain['options']['option']
            dfs.append(pd.DataFrame(opts) if isinstance(opts, list) else pd.DataFrame([opts]))
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
        if not g: continue
        gamma, vega = float(g.get('gamma', 0) or 0), float(g.get('vega', 0) or 0)
        iv = float(g.get('smv_vol') or g.get('mid_iv') or 0)
        if iv > 1.0: iv /= 100.0
        oi, side = int(row.get('open_interest', 0) or 0), (1 if row['option_type'].lower() == 'call' else -1)

        gex = side * gamma * (S**2) * 0.01 * 100 * oi
        vanna_raw = vega / (S * iv) if (S > 0 and iv > 0) else 0
        vanex = side * vanna_raw * S * 0.01 * 100 * oi
        
        res.append({
            "strike": row['strike'], 
            "expiry": row['expiration_date'], 
            "gex": gex, 
            "vanex": vanex, 
            "gamma": gamma * side * oi, # Raw dealer gamma
            "type": row['option_type'].lower(), 
            "oi": oi
        })
    return pd.DataFrame(res)

def find_gamma_flip(df):
    if df.empty: return None
    strike_sums = df.groupby('strike')['gex'].sum().sort_index()
    for i in range(len(strike_sums) - 1):
        if (strike_sums.iloc[i] < 0 and strike_sums.iloc[i+1] > 0) or (strike_sums.iloc[i] > 0 and strike_sums.iloc[i+1] < 0):
            return strike_sums.index[i] if abs(strike_sums.iloc[i]) < abs(strike_sums.iloc[i+1]) else strike_sums.index[i+1]
    return None

def render_heatmap(df, ticker, S, mode, flip_strike):
    pivot = df.pivot_table(index='strike', columns='expiry', values=mode.lower(), aggfunc='sum')
    pivot = pivot.reindex(sorted(pivot.columns), axis=1).sort_index(ascending=False).fillna(0)
    
    z, x_labs, y_labs = pivot.values, pivot.columns.tolist(), pivot.index.tolist()
    abs_limit = np.max(np.abs(z)) if z.size > 0 else 1.0
    closest_strike = min(y_labs, key=lambda x: abs(x - S))

    fig = go.Figure(data=go.Heatmap(z=z, x=x_labs, y=y_labs, colorscale=CUSTOM_COLORSCALE, zmin=-abs_limit, zmax=abs_limit, zmid=0))

    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z[i, j]
            label = f"${val/1e3:,.0f}K" if abs(val) >= 1e3 else f"${val:.0f}"
            fig.add_annotation(x=exp, y=strike, text=label, showarrow=False, font=dict(color="white" if val < 0 else "black", size=10, family="Arial Black"))

    y_text = [f"➔ <b>{s}</b>" if s == closest_strike else (f"⚠️ <b>{s} FLIP</b>" if s == flip_strike else str(s)) for s in y_labs]

    fig.update_layout(title=f"{ticker} {mode} Matrix", template="plotly_dark", height=700, xaxis=dict(side='top', type='category'), yaxis=dict(ticktext=y_text, tickvals=y_labs))
    return fig

# -------------------------
# Dashboard
# -------------------------
def main():
    st.markdown("<h2 style='text-align:center;'>📊 GEX / VANEX Pro Analytics</h2>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1], vertical_alignment="bottom")
    ticker = c1.text_input("Ticker", value="SPY").upper().strip()
    max_exp = c2.number_input("Expiries", 1, 15, 5)
    s_range = c3.number_input("Strike ±", 5, 500, 25)
    refresh = c4.button("🔄 Refresh Data")

    if refresh:
        st.cache_data.clear()

    @st.fragment(run_every="600s")
    def dashboard_content():
        S, raw_df = fetch_data(ticker, max_exp)
        if S and raw_df is not None:
            df = process_exposure(raw_df, S, s_range)
            if not df.empty:
                flip_strike = find_gamma_flip(df)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Net GEX", f"${df['gex'].sum():,.0f}")
                m2.metric("Net VANEX", f"${df['vanex'].sum():,.0f}")
                m3.metric("Gamma Flip", f"${flip_strike:,.0f}" if flip_strike else "N/A")
                m4.metric("Spot Price", f"${S:,.2f}")

                st.markdown("---")
                col_gex, col_van = st.columns(2)
                with col_gex:
                    st.plotly_chart(render_heatmap(df, ticker, S, "GEX", flip_strike), width="stretch")
                with col_van:
                    st.plotly_chart(render_heatmap(df, ticker, S, "VANEX", flip_strike), width="stretch")

                # --- NEW DIAGNOSTIC TABLE ---
                st.markdown("### 🔍 Top 5 Strikes Closest to Spot")
                
                # Aggregate data by strike
                strike_diag = df.groupby('strike').agg({
                    'gex': [('Call GEX', lambda x: x[df.loc[x.index, 'type'] == 'call'].sum()),
                            ('Put GEX', lambda x: x[df.loc[x.index, 'type'] == 'put'].sum()),
                            ('Net GEX', 'sum')],
                    'vanex': [('Call Vanna', lambda x: x[df.loc[x.index, 'type'] == 'call'].sum()),
                              ('Put Vanna', lambda x: x[df.loc[x.index, 'type'] == 'put'].sum()),
                              ('Net Vanna', 'sum')],
                    'gamma': [('Call Gamma', lambda x: x[df.loc[x.index, 'type'] == 'call'].sum()),
                              ('Put Gamma', lambda x: x[df.loc[x.index, 'type'] == 'put'].sum()),
                              ('Net Gamma', 'sum')]
                })
                
                # Flatten columns and find 5 closest to spot
                strike_diag.columns = [c[1] for c in strike_diag.columns]
                strike_diag['Dist %'] = ((strike_diag.index - S) / S * 100).round(2)
                closest_strikes = strike_diag.iloc[(strike_diag.index - S).abs().argsort()[:5]].sort_index(ascending=False)

                # Format and Color
                def color_greeks(val):
                    color = '#2ecc71' if val > 0 else '#e74c3c'
                    return f'color: {color}'

                st.dataframe(
                    closest_strikes.style.format("${:,.0f}")
                    .applymap(color_greeks, subset=['Net GEX', 'Net Vanna', 'Net Gamma']),
                    width="stretch"
                )
                
            else: st.warning("No data in range.")
        else: st.error("API Error.")

    dashboard_content()

if __name__ == "__main__":
    main()