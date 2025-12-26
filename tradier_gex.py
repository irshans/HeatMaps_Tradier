def render_heatmap(df, ticker, S):
    # Pivot the data
    pivot = df.pivot_table(index='strike', columns='expiry', values='gex', aggfunc='sum').sort_index(ascending=False).fillna(0)
    z_raw = pivot.values
    x_labs, y_labs = pivot.columns.tolist(), pivot.index.tolist()
    
    # Calculate symmetry for colorscale
    abs_max = np.max(np.abs(z_raw)) if z_raw.size else 1.0
    
    fig = go.Figure(data=go.Heatmap(
        z=z_raw, x=x_labs, y=y_labs, 
        colorscale=CUSTOM_COLORSCALE, zmin=-abs_max, zmax=abs_max, zmid=0,
        colorbar=dict(title="GEX ($)")
    ))

    # Cell Annotations logic remains the same...
    for i, strike in enumerate(y_labs):
        for j, exp in enumerate(x_labs):
            val = z_raw[i, j]
            if abs(val) < 500: continue
            label = f"${val/1e3:,.0f}K"
            t_color = "black" if val > 0 else "white"
            fig.add_annotation(x=exp, y=strike, text=label, showarrow=False,
                               font=dict(color=t_color, size=11, family="Arial"))

    # --- MODIFIED SECTION START ---
    fig.update_layout(
        title=f"{ticker} GEX Heatmap | Spot: ${S:,.2f}", 
        template="plotly_dark", 
        height=len(y_labs) * 25,  # Dynamically scale height so labels aren't crushed
        xaxis=dict(type='category', side='top'),
        yaxis=dict(
            title="Strike",
            tickmode='array',      # Manual tick control
            tickvals=y_labs,       # Show a tick for every strike in data
            ticktext=[f"{s:,.0f}" for s in y_labs], # Format as numbers
            autorange=True
        ),
        margin=dict(l=80, r=60, t=100, b=40)
    )
    # --- MODIFIED SECTION END ---
    
    return fig