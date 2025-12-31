import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

# --- APP CONFIG ---
st.set_page_config(page_title="GEX & VANEX Pro", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    * { font-family: 'Arial', sans-serif !important; }
    .block-container { padding-top: 24px; padding-bottom: 8px; }
    [data-testid="stMetricValue"] { font-size: 20px !important; font-family: 'Arial' !important; }
    h1, h2, h3 { font-size: 18px !important; margin: 10px 0 6px 0 !important; font-weight: bold; }
    hr { margin: 15px 0 !important; }
    [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 10px; }
    /* Formatting the VEX toggle to be compact */
    div[data-testid="stRadio"] > label { font-size: 14px !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- API HELPERS (TRADIER) ---
if "TRADIER_TOKEN" in st.secrets:
    TRADIER_TOKEN = st.secrets["TRADIER_TOKEN"]
else:
    st.error("Please set TRADIER_TOKEN in Streamlit Secrets.")
    st.stop()

BASE_URL = "https://api.tradier.com/v1/"
CUSTOM_COLORSCALE = [[0.00, '#050018'], [0.10, '#260446'], [0.25, '#56117a'], [0.40, '#6E298A'], [0.49, '#783F8F'], [0.50, '#224B8B'], [0.52, '#32A7A7'], [0.65, '#39B481'], [0.80, '#A8D42A'], [0.92, '#FFDF4A'], [1.00, '#F1F50C']]

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
    today = pd.Timestamp.now()
    res = []
    for _, row in df.iterrows():
        g = row.get('greeks')
        if not g: continue
        gamma, vega, delta = float(g.get('gamma', 0) or 0), float(g.get('vega', 0) or 0), float(g.get('delta', 0) or 0)
        iv = float(g.get('smv_vol') or g.get('mid_iv') or 0.2)
        if iv > 1.0: iv /= 100.0
        oi, side, K = int(row.get('open_interest', 0) or 0), (1 if row['option_type'].lower() == 'call' else -1), float(row['strike'])
        gex = side * gamma * (S**2) * 0.01 * 100 * oi
        vanna_raw = (vega * delta) / (S * max(iv, 0.05))
        vanex_raw = side * vanna_raw * 100 * oi
        tte = max((pd.to_datetime(row["expiration_date"]) - today).days, 0)
        vanex_dealer = -vanna_raw * S * 100 * oi * np.exp(-tte / 30)
        dex = -side * delta * 100 * oi
        res.append({"strike": K, "expiry": row['expiration_date'], "gex": gex, "vanex": vanex_raw, "vanex_dealer": vanex_dealer, "dex": dex, "type": row['option_type'].lower(), "oi": oi})
    return pd.DataFrame(res)

def render_heatmap(df, ticker, S, mode, flip_strike, vanex_type='dealer'):
    val_field = ('vanex' if vanex_type == 'raw' else 'vanex_dealer') if mode == "VEX" else "gex"
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_field, aggfunc='sum').sort_index(ascending=False).fillna(0)
    z, x_labs, y_labs = pivot.values, pivot.columns.tolist(), pivot.index.tolist()
    abs_limit = np.max(np.abs(z)) if z.size > 0 else 1.0
    
    # Highest absolute star logic
    max_pos = np.unravel_index(np.argmax(np.abs(z)), z.shape) if z.size > 0 else None

    fig = go.Figure(data=go.Heatmap(z=z, x=x_labs, y=y_labs, colorscale=CUSTOM_COLORSCALE, zmin=-abs_limit, zmax=abs_limit, zmid=0, ygap=0, xgap=1))
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z[i, j]
            label = f"${val/1e3:,.0f}K" if abs(val) >= 1e3 else f"${val:.0f}"
            if max_pos and (i, j) == max_pos: label += " ⭐"
            fig.add_annotation(x=exp, y=strike, text=label, showarrow=False, font=dict(color="white" if val < 0 else "black", size=9, family="Arial Black"))

    y_t = [f"➔ <b>{s}</b>" if abs(s-S)<0.5 else (f"⚠️ <b>{s} FLIP</b>" if s==flip_strike else str(s)) for s in y_labs]
    fig.update_layout(title=f"{ticker} {mode} Matrix", template="plotly_dark", height=600, margin=dict(l=80, r=20, t=40, b=20), xaxis=dict(side='top', type='category'), yaxis=dict(ticktext=y_t, tickvals=y_labs))
    return fig

# --- MAIN APP ---
def main():
    st.markdown("<h2 style='text-align:center;'>📊 GEX / VANEX Pro Analytics</h2>", unsafe_allow_html=True)
    
    # Header Inputs
    h1, h2, h3, h4 = st.columns([1, 1, 1, 1])
    ticker = h1.text_input("Ticker", value="SPY").upper()
    max_exp = h2.number_input("Expiries", 1, 15, 5)
    s_range = h3.number_input("Strike ±", 5, 200, 25)
    if h4.button("🔄 Refresh Data", use_container_width=True): st.cache_data.clear()

    S, raw_df = fetch_data(ticker, max_exp)
    if S and raw_df is not None:
        df = process_exposure(raw_df, S, s_range)
        if not df.empty:
            # Metrics Row
            flip = None
            s_sums = df.groupby('strike')['gex'].sum().sort_index()
            for i in range(len(s_sums)-1):
                if (s_sums.iloc[i] * s_sums.iloc[i+1]) < 0: flip = s_sums.index[i]; break
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Net GEX", f"${df['gex'].sum():,.0f}")
            m2.metric("DEX Hedging", f"{df['dex'].sum()/1e6:.1f}M Shrs")
            m3.metric("Gamma Flip", f"${flip:,.0f}" if flip else "N/A")
            m4.metric("Spot Price", f"${S:,.2f}")

            st.markdown("---")
            # Bar Charts (Restored Walls)
            b1, b2 = st.columns(2)
            df_bar = df[(df['strike'] >= S-15) & (df['strike'] <= S+15)].copy()
            with b1:
                g_s = df_bar.groupby('strike')['gex'].sum()
                fig_g = go.Figure(go.Bar(y=g_s.index, x=g_s.values, orientation='h', marker_color=['#2ecc71' if v>0 else '#e74c3c' for v in g_s.values]))
                fig_g.add_hline(y=S, line_dash="dash", line_color="yellow", annotation_text="SPOT")
                fig_g.update_layout(title="GEX Concentration", template="plotly_dark", height=400)
                st.plotly_chart(fig_g, use_container_width=True)
            with b2:
                oi_s = df_bar.groupby(['strike','type'])['oi'].sum().unstack(fill_value=0)
                fig_oi = go.Figure()
                for t, c in [('call','#3498db'),('put','#e67e22')]:
                    if t in oi_s.columns: fig_oi.add_trace(go.Bar(name=t, y=oi_s.index, x=oi_s[t], orientation='h', marker_color=c))
                fig_oi.update_layout(title="Open Interest Walls", barmode='stack', template="plotly_dark", height=400)
                st.plotly_chart(fig_oi, use_container_width=True)

            st.markdown("---")
            # Heatmaps Row
            col_gex, col_vex = st.columns(2)
            with col_gex:
                st.markdown("### GEX Matrix")
                st.plotly_chart(render_heatmap(df, ticker, S, "GEX", flip), use_container_width=True)
            
            with col_vex:
                # Toggle placed right above the VEX Heatmap as requested
                v_head, v_toggle = st.columns([1, 1])
                v_head.markdown("### VEX Matrix")
                vex_mode = v_toggle.radio("VEX Source:", options=['dealer', 'raw'], horizontal=True, label_visibility="collapsed")
                st.plotly_chart(render_heatmap(df, ticker, S, "VEX", flip, vex_mode), use_container_width=True)

            # Full Diagnostic Table Restored
            st.markdown("### 🔍 Full Strike Diagnostics")
            diag = pd.DataFrame({
                'Net GEX': df.groupby('strike')['gex'].sum(),
                'Dealer Delta': df.groupby('strike')['dex'].sum(),
                'Vanna (Dlr)': df.groupby('strike')['vanex_dealer'].sum(),
                'Call OI': df[df['type']=='call'].groupby('strike')['oi'].sum(),
                'Put OI': df[df['type']=='put'].groupby('strike')['oi'].sum()
            }).fillna(0)
            diag['Dist %'] = ((diag.index - S)/S*100).round(2)
            st.dataframe(diag.sort_index(ascending=False).style.format("${:,.0f}"), use_container_width=True)
            
    else: st.error("No data found for this ticker.")

if __name__ == "__main__": main()