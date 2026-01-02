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
        
        # Standard GEX calculation
        gex = side * gamma * (S**2) * 0.01 * 100 * oi
        
        # VEX calculations (both raw and dealer-adjusted)
        iv_eff = max(iv, 0.05)  # Use minimum 5% IV
        vanna_raw = (vega * delta) / (S * iv_eff)
        
        # Raw VANEX (customer perspective)
        vanex_raw = side * vanna_raw * 100 * oi
        
        # Dealer VANEX (with time decay and sign flip)
        expiry = pd.to_datetime(row["expiration_date"])
        tte = max((expiry - today).days, 0)
        time_weight = np.exp(-tte / 30)  # Exponential decay with 30-day half-life
        vanex_dealer = -vanna_raw * S * 100 * oi * time_weight
        
        # DEX: Positive for net long delta exposure dealers must hedge
        dex = -side * delta * 100 * oi
        
        # Validate calculations
        if not np.isfinite([gex, vanex_raw, vanex_dealer, dex]).all():
            continue
        
        res.append({
            "strike": K,
            "expiry": row['expiration_date'],
            "gex": gex,
            "vanex": vanex_raw,
            "vanex_dealer": vanex_dealer,
            "dex": dex,
            "gamma": gamma * side * oi,
            "type": row['option_type'].lower(),
            "oi": oi
        })
    
    return pd.DataFrame(res)

def smart_fill_strikes(df, S):
    """Auto-detect strike interval and fill missing strikes for uniform heatmap"""
    if df.empty:
        return df, 5, 25  # Return defaults
    
    # Detect actual interval in data
    strikes = sorted(df['strike'].unique())
    if len(strikes) > 1:
        # Calculate most common interval
        diffs = np.diff(strikes)
        common_interval = np.median(diffs)
    else:
        common_interval = 5
    
    # Round to nearest standard interval and set recommended strike range
    if common_interval <= 1:
        interval = 1
        recommended_range = 25
    elif common_interval <= 2.5:
        interval = 2.5
        recommended_range = 45
    else:
        interval = 5
        recommended_range = 80
    
    st.info(f"📊 Detected strike interval: ${interval} | Recommended range: ±${recommended_range}")
    
    # Create complete strike range
    min_strike = df['strike'].min()
    max_strike = df['strike'].max()
    
    # Adjust min/max to align with interval
    min_strike = np.floor(min_strike / interval) * interval
    max_strike = np.ceil(max_strike / interval) * interval
    
    all_strikes = np.arange(min_strike, max_strike + interval, interval)
    all_expiries = sorted(df['expiry'].unique())
    
    # Create full grid with strike, expiry, AND type
    full_data = []
    for strike in all_strikes:
        for expiry in all_expiries:
            # Check if this strike-expiry combination exists in original data
            existing = df[(df['strike'] == strike) & (df['expiry'] == expiry)]
            
            if not existing.empty:
                # Keep original data
                full_data.append(existing)
            else:
                # Add zero-filled rows for both call and put
                for option_type in ['call', 'put']:
                    full_data.append(pd.DataFrame([{
                        'strike': strike,
                        'expiry': expiry,
                        'gex': 0,
                        'vanex': 0,
                        'vanex_dealer': 0,
                        'dex': 0,
                        'gamma': 0,
                        'oi': 0,
                        'type': option_type
                    }]))
    
    df_final = pd.concat(full_data, ignore_index=True) if full_data else df
    
    return df_final, interval, recommended_range

def find_gamma_flip(df):
    if df.empty: return None
    strike_sums = df.groupby('strike')['gex'].sum().sort_index()
    for i in range(len(strike_sums) - 1):
        if (strike_sums.iloc[i] * strike_sums.iloc[i+1]) < 0:
            return strike_sums.index[i] if abs(strike_sums.iloc[i]) < abs(strike_sums.iloc[i+1]) else strike_sums.index[i+1]
    return None

def render_heatmap(df, ticker, S, mode, flip_strike, vanex_type='dealer'):
    # Determine which vanex field to use
    vanex_field = 'vanex' if vanex_type == 'raw' else 'vanex_dealer'
    value_field = vanex_field if mode.upper() == "VEX" else mode.lower()
    
    pivot = df.pivot_table(index='strike', columns='expiry', values=value_field, aggfunc='sum')
    pivot = pivot.reindex(sorted(pivot.columns), axis=1).sort_index(ascending=False).fillna(0)
    
    z, x_labs, y_labs = pivot.values, pivot.columns.tolist(), pivot.index.tolist()
    abs_limit = np.max(np.abs(z)) if z.size > 0 else 1.0
    closest_strike = min(y_labs, key=lambda x: abs(x - S))
    
    # Find highest absolute value for star marker
    max_abs_val = 0
    max_abs_pos = None
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            if abs(z[i, j]) > max_abs_val:
                max_abs_val = abs(z[i, j])
                max_abs_pos = (i, j)

    fig = go.Figure(data=go.Heatmap(
        z=z, 
        x=x_labs, 
        y=y_labs, 
        colorscale=CUSTOM_COLORSCALE, 
        zmin=-abs_limit, 
        zmax=abs_limit, 
        zmid=0,
        ygap=0,
        xgap=1
    ))

    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z[i, j]
            label = f"${val/1e3:,.0f}K" if abs(val) >= 1e3 else f"${val:.0f}"
            
            # Add star to highest absolute value
            if max_abs_pos and (i, j) == max_abs_pos:
                label += " ⭐"
            
            fig.add_annotation(x=exp, y=strike, text=label, showarrow=False, 
                             font=dict(color="white" if val < 0 else "black", size=9, family="Arial Black"))

    y_text = [f"➔ <b>{s}</b>" if s == closest_strike else (f"⚠️ <b>{s} FLIP</b>" if s == flip_strike else str(s)) for s in y_labs]

    # Simple title without mode suffix for VEX
    title = f"{ticker} {mode} Matrix"
    
    fig.update_layout(
        title=title, 
        template="plotly_dark", 
        height=650, 
        margin=dict(l=80, r=20, t=80, b=20),
        xaxis=dict(side='top', type='category'),
        yaxis=dict(ticktext=y_text, tickvals=y_labs)
    )
    return fig

# -------------------------
# Main Page
# -------------------------
def main():
    st.markdown("<h2 style='text-align:center;'>📊 GEX / VANEX Pro Analytics</h2>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1], vertical_alignment="bottom")
    ticker = c1.text_input("Ticker", value="SPY").upper().strip()
    max_exp = c2.number_input("Expiries", 1, 15, 5)
    
    # Dynamic default for strike range based on ticker
    if 'default_strike_range' not in st.session_state:
        st.session_state.default_strike_range = 80 if ticker == "SPX" else 25
    
    # Update default if ticker changes to/from SPX
    if ticker == "SPX" and st.session_state.default_strike_range == 25:
        st.session_state.default_strike_range = 80
    elif ticker != "SPX" and st.session_state.default_strike_range == 80:
        st.session_state.default_strike_range = 25
    
    s_range = c3.number_input("Strike ±", 5, 500, st.session_state.default_strike_range)
    refresh = c4.button("🔄 Refresh Data")

    if refresh: 
        st.cache_data.clear()
        st.session_state.trigger_refresh = True
    
    # Initialize trigger if not exists
    if 'trigger_refresh' not in st.session_state:
        st.session_state.trigger_refresh = True

    @st.fragment(run_every="600s")
    def dashboard_content():
        # Only fetch data if refresh was triggered
        if not st.session_state.trigger_refresh:
            st.info("👆 Adjust settings above and click 'Refresh Data' to load")
            return
        
        S, raw_df = fetch_data(ticker, max_exp)
        if S and raw_df is not None:
            df = process_exposure(raw_df, S, s_range)
            if not df.empty:
                # Apply smart strike filling for uniform heatmap
                df, interval, recommended_range = smart_fill_strikes(df, S)
                
                # Update session state with recommended range for next refresh
                st.session_state.default_strike_range = recommended_range
                
                flip_strike = find_gamma_flip(df)
                total_dex = df['dex'].sum()
                
                # Determine market regime
                net_gex = df['gex'].sum()
                if flip_strike:
                    if S > flip_strike:
                        regime = "📈 Above Flip"
                        regime_desc = "Positive Gamma Zone"
                    else:
                        regime = "📉 Below Flip"
                        regime_desc = "Negative Gamma Zone"
                else:
                    if net_gex > 0:
                        regime = "📈 Positive Regime"
                        regime_desc = "Stable"
                    else:
                        regime = "📉 Negative Regime"
                        regime_desc = "Volatile"
                
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Net GEX", f"${df['gex'].sum():,.0f}")
                m2.metric("Dealer Delta (DEX)", f"{total_dex/1e6:.1f}M Shrs", delta=f"{'Short' if total_dex < 0 else 'Long'} Hedged")
                m3.metric("Gamma Flip", f"${flip_strike:,.0f}" if flip_strike else "N/A")
                m4.metric("Spot Price", f"${S:,.2f}")
                
                # Hedging Pressure Gauge
                if net_gex > 0:
                    pressure_status = "🟢 Stabilizers"
                    pressure_desc = "Pos Gamma"
                else:
                    pressure_status = "🔴 Forced Sellers"
                    pressure_desc = "Neg Gamma"
                m5.metric("Hedging Pressure", pressure_status, delta=pressure_desc)
                
                # Market Regime
                m6.metric("Market Regime", regime, delta=regime_desc)

                st.markdown("---")
                
                # --- BAR CHARTS ---
                col_bar1, col_bar2 = st.columns(2)
                
                # Dynamic bar range - special handling for SPX
                if ticker == "SPX":
                    bar_range = 50  # Show ±$50 for SPX (about 10 strikes at $5 intervals)
                elif interval >= 5:
                    bar_range = 20  # Other $5 interval tickers
                else:
                    bar_range = 10  # $1-2.5 interval tickers
                    
                df_bar = df[(df['strike'] >= S - bar_range) & (df['strike'] <= S + bar_range) & (df['oi'] > 0)].copy()
                
                if not df_bar.empty:
                    # Find floor and ceiling strikes
                    df_bar_real = df_bar[df_bar['type'] != 'filled']
                    
                    call_oi = df_bar_real[df_bar_real['type'] == 'call'].groupby('strike')['oi'].sum()
                    put_oi = df_bar_real[df_bar_real['type'] == 'put'].groupby('strike')['oi'].sum()
                    
                    ceiling_strike = call_oi.idxmax() if not call_oi.empty else None
                    floor_strike = put_oi.idxmax() if not put_oi.empty else None
                    
                    # A) GEX Concentrations by Strike - HORIZONTAL BARS
                    with col_bar1:
                        gex_by_strike = df_bar.groupby('strike')['gex'].sum().sort_index(ascending=True)
                        
                        fig_gex = go.Figure()
                        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in gex_by_strike.values]
                        
                        fig_gex.add_trace(go.Bar(
                            y=gex_by_strike.index,
                            x=gex_by_strike.values,
                            orientation='h',
                            marker_color=colors,
                            text=[f"${v/1e6:.2f}M" if abs(v) >= 1e6 else f"${v/1e3:.0f}K" for v in gex_by_strike.values],
                            textposition='outside',
                            textfont=dict(size=11, family="Arial Black"),
                            hovertemplate='Strike: $%{y}<br>GEX: $%{x:,.0f}<extra></extra>'
                        ))
                        
                        # Add horizontal line for spot price
                        fig_gex.add_hline(y=S, line_dash="dash", line_color="yellow", line_width=2)
                        
                        # Add floor and ceiling lines
                        if floor_strike:
                            fig_gex.add_hline(y=floor_strike, line_dash="dot", line_color="#e67e22", line_width=2)
                        if ceiling_strike:
                            fig_gex.add_hline(y=ceiling_strike, line_dash="dot", line_color="#3498db", line_width=2)
                        
                        # Create custom y-axis labels with arrow for spot, and markers for floor/ceiling
                        y_labels = []
                        for s in gex_by_strike.index:
                            if abs(s - S) < 0.01:
                                y_labels.append(f"➔ <b>${s:.2f}</b>")
                            elif ceiling_strike and abs(s - ceiling_strike) < 0.01:
                                y_labels.append(f"🔵 ${s:.2f} CEIL")
                            elif floor_strike and abs(s - floor_strike) < 0.01:
                                y_labels.append(f"🟠 ${s:.2f} FLOOR")
                            else:
                                y_labels.append(f"${s:.2f}")
                        
                        fig_gex.update_layout(
                            title=f"GEX Concentration (±${bar_range})",
                            template="plotly_dark",
                            height=350,
                            showlegend=False,
                            yaxis=dict(
                                title="Strike Price",
                                ticktext=y_labels,
                                tickvals=gex_by_strike.index
                            ),
                            xaxis_title="Gamma Exposure ($)",
                            margin=dict(l=110, r=80, t=60, b=20)
                        )
                        
                        st.plotly_chart(fig_gex, use_container_width=True)
                    
                    # B) Options Inventory (OI) by Strike - HORIZONTAL BARS
                    with col_bar2:
                        oi_by_strike = df_bar_real.groupby(['strike', 'type'])['oi'].sum().unstack(fill_value=0)
                        
                        fig_oi = go.Figure()
                        
                        if 'call' in oi_by_strike.columns:
                            fig_oi.add_trace(go.Bar(
                                name='Calls',
                                y=oi_by_strike.index,
                                x=oi_by_strike['call'],
                                orientation='h',
                                marker_color='#3498db',
                                text=[f"{int(v):,}" for v in oi_by_strike['call'].values],
                                textposition='outside',
                                textfont=dict(size=11, family="Arial Black"),
                                hovertemplate='Strike: $%{y}<br>Call OI: %{x:,}<extra></extra>'
                            ))
                        
                        if 'put' in oi_by_strike.columns:
                            fig_oi.add_trace(go.Bar(
                                name='Puts',
                                y=oi_by_strike.index,
                                x=oi_by_strike['put'],
                                orientation='h',
                                marker_color='#e67e22',
                                text=[f"{int(v):,}" for v in oi_by_strike['put'].values],
                                textposition='outside',
                                textfont=dict(size=11, family="Arial Black"),
                                hovertemplate='Strike: $%{y}<br>Put OI: %{x:,}<extra></extra>'
                            ))
                        
                        # Add horizontal line for spot price
                        fig_oi.add_hline(y=S, line_dash="dash", line_color="yellow", line_width=2)
                        
                        # Add floor and ceiling lines
                        if floor_strike:
                            fig_oi.add_hline(y=floor_strike, line_dash="dot", line_color="#e67e22", line_width=2)
                        if ceiling_strike:
                            fig_oi.add_hline(y=ceiling_strike, line_dash="dot", line_color="#3498db", line_width=2)
                        
                        # Create custom y-axis labels with arrow for spot, and markers for floor/ceiling
                        y_labels_oi = []
                        for s in oi_by_strike.index:
                            if abs(s - S) < 0.01:
                                y_labels_oi.append(f"➔ <b>${s:.2f}</b>")
                            elif ceiling_strike and abs(s - ceiling_strike) < 0.01:
                                y_labels_oi.append(f"🔵 ${s:.2f} CEIL")
                            elif floor_strike and abs(s - floor_strike) < 0.01:
                                y_labels_oi.append(f"🟠 ${s:.2f} FLOOR")
                            else:
                                y_labels_oi.append(f"${s:.2f}")
                        
                        fig_oi.update_layout(
                            title=f"Options Inventory (±${bar_range})",
                            template="plotly_dark",
                            height=350,
                            barmode='stack',
                            yaxis=dict(
                                title="Strike Price",
                                ticktext=y_labels_oi,
                                tickvals=oi_by_strike.index
                            ),
                            xaxis_title="Open Interest",
                            margin=dict(l=110, r=80, t=60, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig_oi, use_container_width=True)
                
                st.markdown("---")
                
                col_gex, col_van = st.columns(2)
                with col_gex: 
                    st.plotly_chart(render_heatmap(df, ticker, S, "GEX", flip_strike), use_container_width=True)
                
                with col_van:
                    st.plotly_chart(render_heatmap(df, ticker, S, "VEX", flip_strike, vanex_type='dealer'), use_container_width=True)

                # --- DIAGNOSTIC TABLE ---
                st.markdown("### 🔍 Strike Diagnostics (5 Closest to Spot)")
                
                # Filter out filled strikes (zero OI) for diagnostics
                df_real = df[df['oi'] > 0].copy()
                
                # Create separate dataframes for calls and puts
                df_calls = df_real[df_real['type'] == 'call']
                df_puts = df_real[df_real['type'] == 'put']

                # Aggregate by strike for each type separately
                call_metrics = df_calls.groupby('strike').agg({
                    'gex': 'sum',
                    'vanex': 'sum',
                    'vanex_dealer': 'sum',
                    'gamma': 'sum',
                    'oi': 'sum'
                })
                
                put_metrics = df_puts.groupby('strike').agg({
                    'gex': 'sum',
                    'vanex': 'sum',
                    'vanex_dealer': 'sum',
                    'gamma': 'sum',
                    'oi': 'sum'
                })
                
                net_metrics = df_real.groupby('strike').agg({
                    'gex': 'sum',
                    'vanex': 'sum',
                    'vanex_dealer': 'sum',
                    'gamma': 'sum',
                    'dex': 'sum'
                })

                strike_diag = pd.DataFrame({
                    'Call GEX': call_metrics['gex'],
                    'Put GEX': put_metrics['gex'],
                    'Net GEX': net_metrics['gex'],
                    'Call Vanna (Raw)': call_metrics['vanex'],
                    'Put Vanna (Raw)': put_metrics['vanex'],
                    'Net Vanna (Raw)': net_metrics['vanex'],
                    'Call Vanna (Dlr)': call_metrics['vanex_dealer'],
                    'Put Vanna (Dlr)': put_metrics['vanex_dealer'],
                    'Net Vanna (Dlr)': net_metrics['vanex_dealer'],
                    'Call Gamma': call_metrics['gamma'],
                    'Put Gamma': put_metrics['gamma'],
                    'Net Gamma': net_metrics['gamma'],
                    'Dealer Delta': net_metrics['dex'],
                    'Call OI': call_metrics['oi'],
                    'Put OI': put_metrics['oi']
                }).fillna(0)
                
                strike_diag['Dist %'] = ((strike_diag.index - S) / S * 100).round(2)
                
                # Find floor and ceiling from full data
                if not call_metrics.empty:
                    ceiling_strike_full = call_metrics['oi'].idxmax()
                else:
                    ceiling_strike_full = None
                    
                if not put_metrics.empty:
                    floor_strike_full = put_metrics['oi'].idxmax()
                else:
                    floor_strike_full = None
                
                # Add labels column for floor/ceiling
                strike_diag['Label'] = ''
                if ceiling_strike_full and ceiling_strike_full in strike_diag.index:
                    strike_diag.loc[ceiling_strike_full, 'Label'] = '🔵 CEILING'
                if floor_strike_full and floor_strike_full in strike_diag.index:
                    strike_diag.loc[floor_strike_full, 'Label'] = '🟠 FLOOR'
                
                # Using np.abs to avoid Index object error
                dist_idx = np.abs(strike_diag.index - S).argsort()[:5]
                closest_strikes = strike_diag.iloc[dist_idx].sort_index(ascending=False)

                def color_greeks(val):
                    color = '#2ecc71' if val > 0 else '#e74c3c'
                    return f'color: {color}'

                st.dataframe(
                    closest_strikes.style.format({
                        'Call GEX': '${:,.0f}', 'Put GEX': '${:,.0f}', 'Net GEX': '${:,.0f}',
                        'Call Vanna (Raw)': '${:,.0f}', 'Put Vanna (Raw)': '${:,.0f}', 'Net Vanna (Raw)': '${:,.0f}',
                        'Call Vanna (Dlr)': '${:,.0f}', 'Put Vanna (Dlr)': '${:,.0f}', 'Net Vanna (Dlr)': '${:,.0f}',
                        'Call Gamma': '{:,.2f}', 'Put Gamma': '{:,.2f}', 'Net Gamma': '{:,.2f}',
                        'Dealer Delta': '{:,.0f}', 'Dist %': '{:.2f}%',
                        'Call OI': '{:,.0f}', 'Put OI': '{:,.0f}'
                    }).map(color_greeks, subset=['Net GEX', 'Net Vanna (Raw)', 'Net Vanna (Dlr)', 'Net Gamma', 'Dealer Delta']),
                    use_container_width=True
                )
            else: st.warning("No data in range.")
        else: st.error("API Error.")

    dashboard_content()

if __name__ == "__main__":
    main()