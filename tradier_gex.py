import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
import pytz

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
st.set_page_config(page_title="GEX & VANEX Pro", page_icon="📊", layout="wide")

st.markdown("""
<style>
* { font-family: 'Arial', sans-serif !important; }
.block-container { padding-top: 24px; padding-bottom: 8px; }
[data-testid="stMetricValue"] { font-size: 20px !important; }
h1, h2, h3 { font-size: 18px !important; margin: 10px 0 6px 0 !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SECRETS
# -------------------------------------------------
TRADIER_TOKEN = st.secrets.get("TRADIER_TOKEN")
if not TRADIER_TOKEN:
    st.error("Missing TRADIER_TOKEN")
    st.stop()

BASE_URL = "https://api.tradier.com/v1/"

CUSTOM_COLORSCALE = [
    [0.00, '#050018'], [0.10, '#260446'], [0.25, '#56117a'],
    [0.40, '#6E298A'], [0.49, '#783F8F'], [0.50, '#224B8B'],
    [0.52, '#32A7A7'], [0.65, '#39B481'], [0.80, '#A8D42A'],
    [0.92, '#FFDF4A'], [1.00, '#F1F50C']
]

# -------------------------------------------------
# API HELPERS
# -------------------------------------------------
def tradier_get(endpoint, params):
    headers = {
        "Authorization": f"Bearer {TRADIER_TOKEN}",
        "Accept": "application/json"
    }
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def get_market_days():
    open_days = set()
    now = datetime.now()
    for m in [0, 1]:
        dt = now if m == 0 else (now.replace(day=1) + pd.DateOffset(months=1))
        cal = tradier_get("markets/calendar", {"month": dt.month, "year": dt.year})
        try:
            days = cal["calendar"]["days"]["day"]
            if isinstance(days, dict):
                days = [days]
            for d in days:
                if d["status"] == "open":
                    open_days.add(d["date"])
        except:
            pass
    return open_days

# -------------------------------------------------
# DATA FETCH
# -------------------------------------------------
def fetch_data(symbol, max_exp):
    open_days = get_market_days()

    q = tradier_get("markets/quotes", {"symbols": symbol})
    S = float(q["quotes"]["quote"]["last"])

    exp = tradier_get("markets/options/expirations", {"symbol": symbol})
    dates = exp["expirations"]["date"]
    if not isinstance(dates, list):
        dates = [dates]

    expiries = sorted([d for d in dates if d in open_days])[:max_exp]

    frames = []
    for d in expiries:
        chain = tradier_get("markets/options/chains", {
            "symbol": symbol,
            "expiration": d,
            "greeks": "true"
        })
        if chain and chain["options"]:
            opt = chain["options"]["option"]
            frames.append(pd.DataFrame(opt if isinstance(opt, list) else [opt]))

    return S, pd.concat(frames, ignore_index=True) if frames else None

# -------------------------------------------------
# EXPOSURE CALCULATION
# -------------------------------------------------
def process_exposure(df, S, strike_range):
    if df is None or df.empty:
        return pd.DataFrame()

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df = df[(df["strike"] >= S - strike_range) & (df["strike"] <= S + strike_range)]

    rows = []
    today = pd.Timestamp.today()

    for _, r in df.iterrows():
        g = r.get("greeks")
        if not g:
            continue

        gamma = float(g.get("gamma", 0))
        vega = float(g.get("vega", 0))
        delta = float(g.get("delta", 0))
        iv = float(g.get("smv_vol") or g.get("mid_iv") or 0)
        if iv > 1:
            iv /= 100

        oi = int(r.get("open_interest", 0))
        side = 1 if r["option_type"] == "call" else -1

        # ---------------- GEX ----------------
        gex = side * gamma * (S ** 2) * 0.01 * 100 * oi

        # ---------------- VEX ----------------
        iv_eff = max(iv, 0.05)
        vanna_raw = (vega * delta) / (S * iv_eff)

        vanex_raw = side * vanna_raw * 100 * oi

        expiry = pd.to_datetime(r["expiration_date"])
        tte = max((expiry - today).days, 0)
        time_weight = np.exp(-tte / 30)

        vanex_dealer = -vanna_raw * S * 100 * oi * time_weight

        # ---------------- DEX ----------------
        dex = -side * delta * 100 * oi

        rows.append({
            "strike": r["strike"],
            "expiry": r["expiration_date"],
            "gex": gex,
            "vanex_raw": vanex_raw,
            "vanex_dealer": vanex_dealer,
            "dex": dex,
            "type": r["option_type"],
            "oi": oi
        })

    return pd.DataFrame(rows)

# -------------------------------------------------
# HEATMAP
# -------------------------------------------------
def render_heatmap(df, value_col, title):
    pivot = df.pivot_table(
        index="strike",
        columns="expiry",
        values=value_col,
        aggfunc="sum"
    ).sort_index(ascending=False).fillna(0)

    z = pivot.values
    lim = np.max(np.abs(z)) if z.size else 1

    fig = go.Figure(go.Heatmap(
        z=z,
        x=pivot.columns,
        y=pivot.index,
        colorscale=CUSTOM_COLORSCALE,
        zmin=-lim,
        zmax=lim,
        zmid=0
    ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=650,
        margin=dict(l=80, r=20, t=60, b=20),
        xaxis=dict(side="top")
    )

    return fig

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    st.markdown("<h2 style='text-align:center;'>📊 GEX / VEX Analytics</h2>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    symbol = c1.text_input("Ticker", "SPY")
    max_exp = c2.number_input("Expiries", 1, 10, 5)
    strike_range = c3.number_input("Strike ±", 10, 200, 50)

    S, raw = fetch_data(symbol, max_exp)
    df = process_exposure(raw, S, strike_range)

    if df.empty:
        st.warning("No data")
        return

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            render_heatmap(df, "gex", f"{symbol} GEX"),
            use_container_width=True
        )

    with col2:
        # 🔘 VEX TOGGLE (DEFAULT = DEALER / SKYLIT)
        vex_mode = st.radio(
            "VEX Calculation",
            ["Dealer (Skylit-style)", "Raw (Sensitivity)"],
            index=0,
            horizontal=True
        )

        vex_col = "vanex_dealer" if vex_mode.startswith("Dealer") else "vanex_raw"
        vex_title = f"{symbol} {vex_mode} VEX"

        st.plotly_chart(
            render_heatmap(df, vex_col, vex_title),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
