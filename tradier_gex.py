import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# --- APP CONFIG ---
st.set_page_config(page_title="GEX Pro 2025", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    * { font-family: 'Arial', sans-serif !important; }
    .block-container { padding-top: 24px; padding-bottom: 8px; }
    [data-testid="stMetricValue"] { font-size: 22px !important; font-family: 'Arial' !important; }
    h1, h2, h3 { font-size: 18px !important; margin: 10px 0 6px 0 !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- API TOKEN ---
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
    if not quote_data or 'quotes' not in quote_data:
        st.error("Failed to fetch quote.")
        return None, None

    quote = quote_data['quotes']['quote']
    S = float(quote['last']) if isinstance(quote, dict) else float(quote[0]['last'])

    exp_data = tradier_get("markets/options/expirations", {"symbol": ticker, "includeAllRoots": "true"})
    if not exp_data or 'expirations' not in exp_data:
        st.error("No expirations found.")
        return S, None

    all_exps = exp_data['expirations']['date']
    if not isinstance(all_exps, list):
        all_exps = [all_exps]

    valid_exps = [exp for exp in all_exps if exp in open_days][:max_exp]

    if not valid_exps:
        st.warning("No valid trading day expirations found.")
        return S, None

    dfs = []
    prog = st.progress(0, text="Fetching option chains...")
    for i, exp in enumerate(valid_exps):
        prog.text(f"Fetching {exp} ({i+1}/{len(valid_exps)})")
        chain = tradier_get("markets/options/chains", {"symbol": ticker, "expiration": exp, "greeks": "true"})
        if chain and 'options' in chain and chain['options'] and chain['options']['option']:
            opts = chain['options']['option']
            df_chain = pd.DataFrame(opts) if isinstance(opts, list) else pd.DataFrame([opts])
            dfs.append(df_chain)
        prog.progress((i + 1) / len(valid_exps))
    prog.empty()

    if not dfs:
        return S, None
    return S, pd.concat(dfs, ignore_index=True)

def process_exposure(df, S, s_range):
    if df is None or df.empty:
        return pd.DataFrame()

    df["strike"] = pd.to_numeric(df["strike"], errors='coerce')
    df = df[(df["strike"] >= S - s_range) & (df["strike"] <= S + s_range)].copy()

    res = []
    for _, row in df.iterrows():
        g = row.get('greeks')
        if not g or not isinstance(g, dict):
            continue
        gamma = float(g.get('gamma', 0) or 0)
        vega = float(g.get('vega', 0) or 0)
        oi = int(row.get('open_interest', 0) or 0)
        op_type = row['option_type'].lower()

        side = 1 if op_type == 'call' else -1

        res.append({
            "strike": row['strike'],
            "expiry": row['expiration_date'],
            "gex": side * gamma * (S**2) * 0.01 * CONTRACT_SIZE * oi,
            "vex": side * vega * 0.01 * CONTRACT_SIZE * oi,
            "type": op_type,
            "oi": oi
        })

    return pd.DataFrame(res)

# -------------------------
# Visualizations
# -------------------------
def render_heatmap(df, ticker, S):
    pivot = df.pivot_table(index='strike', columns='expiry', values='gex', aggfunc='sum').sort_index(ascending=False).fillna(0)
    z_raw = pivot.values
    x_labs, y_labs = pivot.columns.tolist(), pivot.index.tolist()

    abs_limit = np.max(np.abs(z_raw)) if z_raw.size > 0 else 1.0
    closest_strike = min(y_labs, key=lambda x: abs(x - S))

    fig = go.Figure(data=go.Heatmap(
        z=z_raw, x=x_labs, y=y_labs,
        colorscale=CUSTOM_COLORSCALE, zmin=-abs_limit, zmax=abs_limit, zmid=0,
        colorbar=dict(title="GEX ($)", tickfont=dict(family="Arial"))
    ))

    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < 500:
                continue
            star = " ★" if np.isclose(abs(val), abs_limit) else ""
            label = f"${val/1e3:,.0f}K{star}"
            t_color = "black" if val >= 0 else "white"
            fig.add_annotation(
                x=exp, y=strike, text=label, showarrow=False,
                font=dict(color=t_color, size=12, family="Arial")
            )

    calc_height = max(600, len(y_labs) * 25)
    ticktext = [f"➔ <b>{s:,.0f}</b>" if s == closest_strike else f"{s:,.0f}" for s in y_labs]

    fig.update_layout(
        title=f"{ticker} GEX Matrix | Spot: ${S:,.2f}",
        template="plotly_dark",
        height=calc_height,
        font=dict(family="Arial"),
        xaxis=dict(type='category', side='top', tickfont=dict(size=12)),
        yaxis=dict(
            title="Strike",
            tickmode='array',
            tickvals=y_labs,
            ticktext=ticktext,
            tickfont=dict(size=12)
        )
    )
    return fig

def render_gamma_bar(df, S):
    if df.empty:
        return None

    agg = df.groupby('strike')['gex'].sum().sort_index()
    if agg.empty:
        return None

    strikes = agg.index.tolist()
    gex_vals = agg.values

    flip_strike = None
    for i in range(len(gex_vals) - 1):
        if gex_vals[i] * gex_vals[i + 1] < 0:
            flip_strike = np.interp(0, [gex_vals[i], gex_vals[i+1]], [strikes[i], strikes[i+1]])
            break

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=strikes,
        y=gex_vals,
        marker_color=['#F1F50C' if v > 0 else '#56117a' for v in gex_vals],
        name="Net GEX"
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    if flip_strike:
        fig.add_vline(x=flip_strike, line_dash="dash", line_color="white",
                      annotation_text="Gamma Flip", annotation_position="top left")

    fig.update_layout(
        title="Structural Gamma Walls (Total GEX by Strike)",
        template="plotly_dark",
        height=450,
        font=dict(family="Arial"),
        xaxis=dict(title="Strike Price", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="Total Net GEX ($)", gridcolor='rgba(255,255,255,0.1)'),
        margin=dict(l=80, r=40, t=80, b=40),
        bargap=0.2
    )

    return fig

# -------------------------
# Main App
# -------------------------
def main():
    st.markdown("<h2 style='text-align:center;'>📊 GEX Pro Analytics</h2>", unsafe_allow_html=True)

    # --- Auto-refresh controls ---
    refresh_col1, refresh_col2, refresh_col3 = st.columns([1, 1, 2])
    auto_refresh = refresh_col1.toggle("Auto‑Refresh", value=False)
    refresh_interval = 60  # seconds

    if auto_refresh:
        st_autorefresh(interval=refresh_interval * 1000, key="gex_refresh")

        # Countdown timer
        if "countdown" not in st.session_state:
            st.session_state.countdown = refresh_interval

        st.session_state.countdown -= 1
        if st.session_state.countdown <= 0:
            st.session_state.countdown = refresh_interval

        refresh_col2.write(f"⏳ Refreshing in **{st.session_state.countdown}s**")

    # Last updated timestamp
    refresh_col3.write(f"🕒 Last updated: **{datetime.now().strftime('%H:%M:%S')}**")

    # --- Inputs ---
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 0.8], vertical_alignment="bottom")

    ticker = c1.text_input("Ticker", value="SPX").upper().strip()
    max_exp = c2.number_input("Expiries", min_value=1, max_value=15, value=6)
    s_range = c3.number_input("Strike ±", min_value=5, max_value=500, value=80 if ticker == "SPX" else 25)
    run = c4.button("Run", type="primary")

    if run or auto_refresh:
        with st.spinner("Fetching data from Tradier..."):
            S, raw_df = fetch_data(ticker, int(max_exp))

        if S is not None and raw_df is not None and not raw_df.empty:
            df = process_exposure(raw_df, S, s_range)
            if not df.empty:
                # Top Metrics
                m1, m2, m3, m4, m5 = st.columns(5)
                total_gex = df['gex'].sum()
                total_vex = df['vex'].sum()
                calls_oi = df[df['type'] == 'call']['oi'].sum()
                puts_oi = df[df['type'] == 'put']['oi'].sum()
                cp_ratio = calls_oi / puts_oi if puts_oi > 0 else 0

                m1.metric("Net GEX", f"${total_gex/1e9:,.2f}B")
                m2.metric("Net VEX", f"${total_vex/1e6:,.1f}M")
                m3.metric("Total Calls OI", f"{calls_oi:,.0f}")
                m4.metric("Total Puts OI", f"{puts_oi:,.0f}")
                m5.metric("C/P Ratio", f"{cp_ratio:.2f}")

                # Charts
                st.plotly_chart(render_heatmap(df, ticker, S), use_container_width=True)
                bar_fig = render_gamma_bar(df, S)
                if bar_fig:
                    st.plotly_chart(bar_fig, use_container_width=True)
                else:
                    st.info("Not enough data for bar chart.")
            else:
                st.warning("No options data in selected strike range.")
        else:
            st.error("Failed to fetch data. Check ticker or API connection.")

if __name__ == "__main__":
    main()