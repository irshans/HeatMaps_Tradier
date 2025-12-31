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
    [data-testid="stMetricValue"] { font-size: 20px !important; font-family: 'Arial' !important; }
    h1, h2, h3 { font-size: 18px !important; margin: 10px 0 6px 0 !important; font-weight: bold; }
    hr { margin: 15px 0 !important; }
    [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 10px; }
    /* Hide the radio label to keep the top bar clean */
    div[data-testid="stRadio"] > label { display: none; }
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
    
    today = pd.Timestamp.now()
    
    res = []
    for _, row in df.iterrows():
        g = row.get('greeks')
        if not g: continue
        gamma = float(g.get('gamma', 0) or 0)
        vega = float(g.get('vega', 0) or 0)
        delta = float(g.get('delta', 0) or 0)
        iv = float(g.get('smv_vol') or g.get('mid_iv') or 0)
        if iv > 1.0: iv /= 100.0
        oi = int(row.get('open_interest', 0) or 0)
        side = 1 if row['option_type'].lower() == 'call' else -1
        K = float(row['strike'])
        
        gex = side * gamma * (S**2) * 0.01 * 100 * oi
        iv_eff = max(iv, 0.05)
        vanna_raw = (vega * delta) / (S * iv_eff)
        vanex_raw = side * vanna_raw * 100 * oi
        
        expiry = pd.to_datetime(row["expiration_date"])
        tte = max((expiry - today).days, 0)
        time_weight = np.exp(-tte / 30)
        vanex_dealer = -vanna_raw * S * 100 * oi * time_weight
        dex = -side * delta * 100 * oi
        
        if not np.isfinite([gex, vanex_raw, vanex_dealer, dex]).all(): continue
        
        res.append({
            "strike": K, "expiry": row['expiration_date'], "gex": gex,
            "vanex": vanex_raw, "vanex_dealer": vanex_dealer, "dex": dex,
            "gamma": gamma * side * oi, "type": row['option_type'].lower(), "oi": oi
        })
    return pd.DataFrame(res)

def smart_fill_strikes(df, S):
    if df.empty: return df, 5, 25
    strikes = sorted(df['strike'].unique())
    common_interval = np.median(np.diff(strikes)) if len(strikes) > 1 else 5
    
    if common_interval <= 1: interval, recommended_range = 1, 25
    elif common_interval <= 2.5: interval, recommended_range = 2.5, 45
    else: interval, recommended_range = 5, 80
    
    st.info(f"📊 Detected strike interval: ${interval} | Recommended range: ±${recommended_range}")
    
    min_strike = np.floor(df['strike'].min() / interval) * interval
    max_strike = np.ceil(df['strike'].max() / interval) * interval
    all_strikes = np.arange(min_strike, max_strike + interval, interval)
    all_expiries = sorted(df['expiry'].unique())
    
    full_index = pd.MultiIndex.from_product([all_strikes, all_expiries], names=['strike', 'expiry'])
    df_agg = df.groupby(['strike', 'expiry']).agg({
        'gex': 'sum', 'vanex': 'sum', 'vanex_dealer': 'sum', 'dex': 'sum', 'gamma': 'sum', 'oi': 'sum'
    }).reindex(full_index, fill_value=0).reset_index()
    
    df_with_type = df.groupby(['strike', 'expiry', 'type']).agg({'oi': 'sum'}).reset_index()
    df_final = df_agg.merge(df_with_type, on=['strike', 'expiry'], how='left', suffixes=('', '_typed'))
    df_final['type'] = df_final['type'].fillna('filled')
    df_final['oi'] = df_final['oi_typed'].fillna(df_final['oi'])
    return df_final.drop(columns=['oi_typed'], errors='ignore'), interval, recommended_range

def find_gamma_flip(df):
    if df.empty: return None
    strike_sums = df.groupby('strike')['gex'].sum().sort_index()
    for i in range(len(strike_sums) - 1):
        if (strike_sums.iloc[i] * strike_sums.iloc[i+1]) < 0:
            return strike_sums.index[i] if abs(strike_sums.iloc[i]) < abs(strike_sums.iloc[i+1]) else strike_sums.index[i+1]
    return None

def render_heatmap(df, ticker, S, mode, flip_strike, vanex_type='dealer'):
    val_field = ('vanex' if vanex_type == 'raw' else 'vanex_dealer') if mode.upper() == "VEX" else mode.lower()
    pivot = df.pivot_table(index='strike', columns='expiry', values=val_field, aggfunc='sum').sort_index(ascending=False).fillna(0)
    z, x_labs, y_labs = pivot.values, pivot.columns.tolist(), pivot.index.tolist()
    abs_limit = np.max(np.abs(z)) if z.size > 0 else 1.0
    closest_strike = min(y_labs, key=lambda x: abs(x - S))

    # Find highest absolute value for Star (Works for GEX and VEX)
    max_abs_pos = None
    if z.size > 0:
        max_idx = np.unravel_index(np.argmax(np.abs(z), axis=None), z.shape)
        max_abs_pos = max_idx

    fig = go.Figure(data=go.Heatmap(z=z, x=x_labs, y=y_labs, colorscale=CUSTOM_COLORSCALE, zmin=-abs_limit, zmax=abs_limit, zmid=0, ygap=0, xgap=1))

    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z[i, j]
            label = f"${val/1e3:,.0f}K" if abs(val) >= 1e3 else f"${val:.0f}"
            if max_abs_pos and (i, j) == max_abs_pos: label += " ⭐"
            fig.add_annotation(x=exp, y=strike, text=label, showarrow=False, font=dict(color="white" if val < 0 else "black", size=9, family="Arial Black"))

    y_text = [f"➔ <b>{s}</b>" if s == closest_strike else (f"⚠️ <b>{s} FLIP</b>" if s == flip_strike else str(s)) for s in y_labs]
    fig.update_layout(title=f"{ticker} {mode} Matrix" + (f" ({vanex_type})" if mode == "VEX" else ""), template="plotly_dark", height=650, margin=dict(l=80, r=20, t=80, b=20), xaxis=dict(side='top', type='category'), yaxis=dict(ticktext=y_text, tickvals=y_labs))
    return fig

# -------------------------
# Main Page
# -------------------------
def main():
    st.markdown("<h2 style='text-align:center;'>📊 GEX / VANEX Pro Analytics</h2>", unsafe_allow_html=True)
    
    # 1) UI Change: VEX toggle moved to the top bar row
    c1, c2, c_toggle, c3, c4 = st.columns([1, 0.7, 1.2, 0.7, 0.7], vertical_alignment="bottom")
    ticker = c1.text_input("Ticker", value="SPY").upper().strip()
    max_exp = c2.number_input("Expiries", 1, 15, 5)
    
    # Toggle placed right next to "Expiries"
    vex_toggle = c_toggle.radio("VEX Mode", options=['dealer', 'raw'], horizontal=True, key='vex_toggle')
    
    if 'default_strike_range' not in st.session_state: st.session_state.default_strike_range = 25
    s_range = c3.number_input("Strike ±", 5, 500, st.session_state.default_strike_range)
    if c4.button("🔄 Refresh"): st.cache_data.clear()

    @st.fragment(run_every="600s")
    def dashboard_content():
        S, raw_df = fetch_data(ticker, max_exp)
        if S and raw_df is not None:
            df, interval, recommended_range = smart_fill_strikes(process_exposure(raw_df, S, s_range), S)
            if not df.empty:
                st.session_state.default_strike_range = recommended_range
                flip_strike = find_gamma_flip(df)
                total_dex = df['dex'].sum()
                
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Net GEX", f"${df['gex'].sum():,.0f}")
                m2.metric("Dealer Delta (DEX)", f"{total_dex/1e6:.1f}M Shrs", delta=f"{'Short' if total_dex < 0 else 'Long'}")
                m3.metric("Gamma Flip", f"${flip_strike:,.0f}" if flip_strike else "N/A")
                m4.metric("Spot Price", f"${S:,.2f}")
                m5.metric("Pressure", "🟢 Stabilizers" if df['gex'].sum() > 0 else "🔴 Sellers")

                st.markdown("---")
                
                # Full Bar Charts Restored
                col_bar1, col_bar2 = st.columns(2)
                bar_range = 10
                df_bar = df[(df['strike'] >= S - bar_range) & (df['strike'] <= S + bar_range) & (df['oi'] > 0)].copy()
                
                if not df_bar.empty:
                    # GEX Bar Chart
                    with col_bar1:
                        g_strike = df_bar.groupby('strike')['gex'].sum().sort_index()
                        fig_g = go.Figure(go.Bar(y=g_strike.index, x=g_strike.values, orientation='h', marker_color=['#2ecc71' if v > 0 else '#e74c3c' for v in g_strike.values], text=[f"${v/1e6:.1f}M" for v in g_strike.values], textposition='outside'))
                        fig_g.add_hline(y=S, line_dash="dash", line_color="yellow")
                        fig_g.update_layout(title="GEX Concentration", template="plotly_dark", height=380, margin=dict(l=80, r=50))
                        st.plotly_chart(fig_g, use_container_width=True)
                    
                    # OI Bar Chart
                    with col_bar2:
                        oi_s = df_bar[df_bar['type'] != 'filled'].groupby(['strike', 'type'])['oi'].sum().unstack(fill_value=0)
                        fig_o = go.Figure()
                        for t, c in zip(['call', 'put'], ['#3498db', '#e67e22']):
                            if t in oi_s.columns: fig_o.add_trace(go.Bar(name=t.capitalize(), y=oi_s.index, x=oi_s[t], orientation='h', marker_color=c))
                        fig_o.add_hline(y=S, line_dash="dash", line_color="yellow")
                        fig_o.update_layout(title="Options Inventory", barmode='stack', template="plotly_dark", height=380)
                        st.plotly_chart(fig_o, use_container_width=True)

                st.markdown("---")
                
                # Heatmaps with Stars
                col_h1, col_h2 = st.columns(2)
                with col_h1: st.plotly_chart(render_heatmap(df, ticker, S, "GEX", flip_strike), use_container_width=True)
                with col_h2: st.plotly_chart(render_heatmap(df, ticker, S, "VEX", flip_strike, vanex_type=vex_toggle), use_container_width=True)

                # Diagnostic Table Restored
                st.markdown("### 🔍 Strike Diagnostics")
                df_real = df[df['oi'] > 0].copy()
                diag = pd.DataFrame({
                    'Net GEX': df_real.groupby('strike')['gex'].sum(),
                    'Net Vanna (Dlr)': df_real.groupby('strike')['vanex_dealer'].sum(),
                    'Dealer Delta': df_real.groupby('strike')['dex'].sum(),
                    'Call OI': df_real[df_real['type']=='call'].groupby('strike')['oi'].sum(),
                    'Put OI': df_real[df_real['type']=='put'].groupby('strike')['oi'].sum()
                }).fillna(0)
                st.dataframe(diag.sort_index(ascending=False).style.format("${:,.0f}"), use_container_width=True)
            else: st.warning("No data in range.")
        else: st.error("API Error.")

    dashboard_content()

if __name__ == "__main__":
    main()