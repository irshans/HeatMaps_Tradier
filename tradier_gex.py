import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# --- APP CONFIG ---
st.set_page_config(page_title="GEX Pro 2025", page_icon="📊", layout="wide")

# Compact UI styling (exactly from script 2)
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
    .css-1lcbmhc.e1tzin5v0 { gap: 6px; } /* fallback: small column gap */
    
    /* Force header size reduction */
    h1, h2, h3 { font-size: 18px !important; margin: 10px 0 6px 0 !important; }
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

# Exact color scheme from script 2
CUSTOM_COLORSCALE = [
    [0.00, '#050018'],  # deepest purple (extreme negative)
    [0.10, '#260446'],
    [0.25, '#56117a'],
    [0.40, '#6E298A'],
    [0.49, '#783F8F'],  # last purple before center
    [0.50, '#224B8B'],  # explicit center (neutral blue) — separates neg/pos
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
    """Fetch market calendar for current and next year, month by month"""
    from datetime import datetime
    open_days = set()
    
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # Fetch remaining months of current year + all of next year
    for year in [current_year, current_year + 1]:
        start_month = current_month if year == current_year else 1
        for month in range(start_month, 13):
            cal = tradier_get("markets/calendar", {"month": month, "year": year})
            try:
                if cal and 'calendar' in cal and 'days' in cal['calendar']:
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
    
    # Take first max_exp valid open market days to ensure chronological order
    valid_exps = []
    for exp in all_exps:
        if exp in open_days:
            valid_exps.append(exp)
        if len(valid_exps) >= max_exp:
            break
    
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
# GEX & VEX Calculation
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
        vega = float(g.get('vega', 0) or 0)
        oi = int(row.get('open_interest', 0) or 0)
        option_type = row['option_type'].lower()
        
        # Dealer model: Short Calls (-1) / Long Puts (+1)
        dealer_pos = -1 if option_type == 'call' else 1
        
        # GEX Formula (per 1% spot move)
        gex_val = dealer_pos * gamma * (S**2) * 0.01 * CONTRACT_SIZE * oi
        # VEX Formula (per 1% IV move)
        vex_val = dealer_pos * vega * 0.01 * CONTRACT_SIZE * oi
        
        res.append({
            "strike": row['strike'], 
            "expiry": row['expiration_date'],
            "gex": gex_val,
            "vex": vex_val,
            "option_type": option_type,
            "oi": oi
        })
    return pd.DataFrame(res)

def render_gamma_wall_chart(df, S):
    """Create gamma wall bar chart showing net GEX by strike with put/call walls and flip point"""
    if df.empty:
        return None
    
    # Aggregate by strike
    agg = df.groupby('strike')['gex'].sum().sort_index()
    
    # Find key levels
    max_neg_strike = agg[agg == agg.min()].index[0] if agg.min() < 0 else None
    max_pos_strike = agg[agg == agg.max()].index[0] if agg.max() > 0 else None
    
    # Find gamma flip (where GEX crosses zero)
    gamma_flip = None
    strikes = agg.index.tolist()
    for i in range(len(strikes) - 1):
        if agg.iloc[i] < 0 and agg.iloc[i + 1] > 0:
            gamma_flip = strikes[i + 1]
            break
        elif agg.iloc[i] > 0 and agg.iloc[i + 1] < 0:
            gamma_flip = strikes[i]
            break
    
    # Create bar chart
    colors = ['#2563eb' if v < 0 else '#fbbf24' for v in agg.values]
    
    fig = go.Figure(go.Bar(
        x=agg.index,
        y=agg.values,
        marker_color=colors,
        hovertemplate='Strike: $%{x:,.0f}<br>GEX: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add spot price marker
    fig.add_annotation(
        x=S, y=0,
        text="▲ Spot",
        showarrow=False,
        font=dict(color="yellow", size=12),
        yref="paper",
        yshift=-20
    )
    
    # Add Put Wall annotation (largest negative GEX)
    if max_neg_strike:
        fig.add_annotation(
            x=max_neg_strike,
            y=agg[max_neg_strike],
            text="Put Wall",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#2563eb",
            font=dict(color="#2563eb", size=11),
            ax=0,
            ay=-40
        )
    
    # Add Call Wall annotation (largest positive GEX)
    if max_pos_strike:
        fig.add_annotation(
            x=max_pos_strike,
            y=agg[max_pos_strike],
            text="Call Wall",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#fbbf24",
            font=dict(color="#fbbf24", size=11),
            ax=0,
            ay=40
        )
    
    # Add Gamma Flip line
    if gamma_flip:
        fig.add_vline(
            x=gamma_flip,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Gamma Flip: ${gamma_flip:,.0f}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title="Gamma Wall (Net GEX by Strike)",
        template="plotly_dark",
        height=400,
        xaxis=dict(title="Strike", tickformat=",d"),
        yaxis=dict(title="Net GEX ($)", tickformat="$,.2s"),
        showlegend=False
    )
    
    return fig
    pivot = df.pivot_table(index='strike', columns='expiry', values='gex', aggfunc='sum').sort_index(ascending=False).fillna(0)
    
    z_raw = pivot.values
    # Scale for visual but use raw for color mapping
    z_scaled = np.sign(z_raw) * (np.abs(z_raw) ** 0.5)
    
    x_labs = pivot.columns.tolist()
    y_labs = pivot.index.tolist()
    
    if not y_labs:
        return None
    
    closest_strike = min(y_labs, key=lambda x: abs(x - S))
    
    # Build hover text using raw values
    h_text = []
    for i, strike in enumerate(y_labs):
        row = []
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
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
    
    # Symmetric range so zero maps to center exactly
    max_abs = np.max(np.abs(z_raw)) if z_raw.size else 1.0
    if max_abs == 0:
        max_abs = 1.0
    zmin = -max_abs
    zmax = max_abs
    
    # Heatmap using raw z values
    fig = go.Figure(data=go.Heatmap(
        z=z_raw, x=x_labs, y=y_labs, text=h_text, hoverinfo="text",
        colorscale=CUSTOM_COLORSCALE, zmin=zmin, zmax=zmax, zmid=0, showscale=True,
        colorbar=dict(title=dict(text="GEX ($)"), tickformat=",.0s")
    ))

    # Cell annotations (show only above threshold)
    max_abs_val = np.max(np.abs(z_raw)) if z_raw.size else 0
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < 500:
                continue
            prefix = "-" if val < 0 else ""
            txt = f"{prefix}${abs(val)/1e3:,.0f}K"
            if abs(val) == max_abs_val and max_abs_val > 0:
                txt += " ⭐"

            # Use scaled z for text color determination
            cell_val = z_scaled[i, j]
            zmin_s = z_scaled.min() if z_scaled.size else -1
            zmax_s = z_scaled.max() if z_scaled.size else 1
            if zmax_s != zmin_s:
                z_norm = (cell_val - zmin_s) / (zmax_s - zmin_s)
            else:
                z_norm = 0.5
            text_color = "black" if z_norm > 0.55 else "white"

            fig.add_annotation(x=exp, y=strike, text=txt, showarrow=False, 
                             font=dict(color=text_color, size=12, family="Arial"), 
                             xref="x", yref="y")

    # Highlight background for spot strike
    sorted_strikes = sorted(y_labs)
    strike_diffs = np.diff(sorted_strikes) if len(sorted_strikes) > 1 else np.array([sorted_strikes[0] * 0.05])
    padding = (strike_diffs[0] * 0.45) if len(strike_diffs) > 0 else 2.5

    fig.add_shape(type="rect", xref="paper", yref="y", x0=-0.08, x1=1.0, 
                  y0=closest_strike - padding, y1=closest_strike + padding, 
                  fillcolor="rgba(255, 51, 51, 0.25)", line=dict(width=0), layer="below")

    fig.update_layout(
        title=f"{ticker} GEX Exposure Map | Spot: ${S:,.2f} — Dealer: Short Calls / Long Puts", 
        template="plotly_dark", height=900, 
        xaxis=dict(type='category', side='top', tickfont=dict(size=12)), 
        yaxis=dict(title="Strike", tickfont=dict(size=12), autorange=True, 
                   tickmode='array', tickvals=y_labs, 
                   ticktext=[f"<b>{s:,.0f}</b>" if s == closest_strike else f"{s:,.0f}" for s in y_labs]), 
        margin=dict(l=80, r=60, t=100, b=40)
    )
    return fig

# -------------------------
# Main App
# -------------------------
def main():
    # Small centered title (exact match to script 2)
    st.markdown(
        "<div style='text-align:center; margin-top:6px;'><h2 style='font-size:18px; margin:10px 0 6px 0; font-weight:600;'>📊 GEX Pro (Tradier)</h2></div>",
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
                # Calculate metrics
                t_gex = processed["gex"].sum() / 1e9
                t_vex = processed["vex"].sum() / 1e9
                total_calls = processed[processed["option_type"] == "call"]["oi"].sum()
                total_puts = processed[processed["option_type"] == "put"]["oi"].sum()
                call_put_ratio = total_calls / total_puts if total_puts > 0 else 0
                
                # Display metrics in columns
                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                
                p_g = "-" if t_gex < 0 else ""
                p_v = "-" if t_vex < 0 else ""
                
                col_m1.metric("Net Dealer GEX", f"{p_g}${abs(t_gex):,.2f}B")
                col_m2.metric("Net Dealer VEX", f"{p_v}${abs(t_vex):,.2f}B")
                col_m3.metric("Total Calls", f"{total_calls:,.0f}")
                col_m4.metric("Total Puts", f"{total_puts:,.0f}")
                col_m5.metric("Call/Put Ratio", f"{call_put_ratio:.2f}")
                
                # Render heatmap
                fig = render_heatmap(processed, ticker, S)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Render gamma wall chart
                gamma_fig = render_gamma_wall_chart(processed, S)
                if gamma_fig:
                    st.plotly_chart(gamma_fig, use_container_width=True)
            else: 
                st.warning("No data in range. Broaden strike range or check ticker.")
        else: 
            st.error("Fetch failed. Check API token or market hours.")

if __name__ == "__main__":
    main()