import streamlit as st
import pandas as pd
import plotly.express as px
import requests

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="UP Agri Portfolio Optimizer", layout="wide")
st.title("Uttar Pradesh Crop Portfolio Optimizer")
st.markdown("Risk-adjusted crop allocation using Pyomo LP with a downside MAD penalty and an API-driven architecture.")

# --- INTERACTIVE SIDEBAR WITH FORM BATCHING ---
st.sidebar.header("Optimization Parameters")
st.sidebar.markdown("Adjust the constraints below, then click the button to resolve the Pyomo matrix.")

# Wrapping the inputs in a form stops Streamlit from rerunning on every millimeter of slider drag
with st.sidebar.form(key='optimization_form'):
    user_risk = st.slider(
        "Risk Aversion Penalty (λ)", 
        min_value=0.0, max_value=1.0, value=0.50, step=0.05, 
        help="0.0 = Maximize pure profit (High Risk). 1.0 = Maximize safety against droughts (Low Risk)."
    )

    user_water_cost = st.slider(
        "Groundwater Pumping Cost (₹/mm)", 
        min_value=10.0, max_value=50.0, value=25.0, step=1.0,
        help="Increases the financial penalty for planting water-heavy crops like Sugarcane in drought-prone districts."
    )
    
    # The submit button belongs inside the form. The app only updates when this is clicked.
    submit_button = st.form_submit_button(label="🚀 Run Dynamic Simulation")

st.sidebar.divider()
st.sidebar.info("💡 **How it works:** Moving the sliders is now instant and lag-free. The math only runs across the FastAPI backend when you click the submit button.")

API_URL = "http://127.0.0.1:8000/optimize"

# ==========================================
# 2. API CONNECTION & DATA FETCHING
# ==========================================
@st.cache_data(ttl=3600)
def fetch_portfolio_state(risk, water):
    """Sends the UI slider values to the FastAPI backend via POST request."""
    payload = {"risk_aversion": risk, "water_cost": water}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Cannot connect to the Optimization Engine. Ensure the FastAPI server is running in a separate terminal. Error: {e}")
        st.stop()

# When the form is submitted, clear the cache so it forces a fresh API request
if submit_button:
    st.cache_data.clear()

with st.spinner("Injecting mutable constraints & solving matrix..."):
    # This runs on first load (default values) and whenever the form button is clicked
    state = fetch_portfolio_state(user_risk, user_water_cost)

# ==========================================
# 3. DASHBOARD VISUALIZATIONS
# ==========================================

# --- SECTION A: Optimal Allocation (The Treemap Upgrade) ---
st.subheader("1. Recommended Crop Mix")
st.caption("Optimal hectares from the MAD-adjusted LP. Treemaps are mathematically superior to pie charts for comparing hierarchical land allocation.")

allocation_data = [{'Crop': k, 'Hectares': v * 1000} for k, v in state['optimal_portfolio'].items() if v > 0]
df_alloc = pd.DataFrame(allocation_data).sort_values(by='Hectares', ascending=False)

if df_alloc.empty:
    st.warning("The solver returned an empty portfolio. Try adjusting the parameters to be less strict.")
else:
    col_alloc1, col_alloc2 = st.columns([1, 2])
    
    with col_alloc1:
        # Show the raw numbers in a clean table
        st.dataframe(df_alloc.style.format({'Hectares': '{:,.0f}'}), use_container_width=True)
        
    with col_alloc2:
        # THE TREEMAP: Replaces the pie chart (UPDATED FOR DISTINCT COLORS)
        fig_tree = px.treemap(
            df_alloc, 
            path=[px.Constant("Total UP Arable Land"), 'Crop'], 
            values='Hectares',
            color='Crop',  # <--- FIX 1: Color by Crop Name, not by Size
            color_discrete_sequence=px.colors.qualitative.Prism, # <--- FIX 2: Use a distinct, vibrant color palette
            hover_data={'Hectares': ':,.0f'}
        )
        fig_tree.update_traces(textinfo="label+percent parent")
        fig_tree.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
        st.plotly_chart(fig_tree, use_container_width=True)

st.divider()

# --- SECTION B: Historical Back-Test (The Box Plot Upgrade) ---
st.subheader("2. Historical Net-Profit Back-Test")
st.caption("Deterministic replay over 10 historical years. The Box Plot reveals the true distribution of extreme weather shocks.")

history_data = [{'Year': str(k), 'Net_Profit': v} for k, v in state['portfolio_history'].items()]
df_history = pd.DataFrame(history_data).sort_values("Year")

mean_profit = df_history['Net_Profit'].mean()
worst_case = df_history['Net_Profit'].min()
best_case = df_history['Net_Profit'].max()
volatility = df_history['Net_Profit'].std()

# Key Performance Indicators (KPIs)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Expected Mean Profit", f"₹ {mean_profit / 10_000_000:,.2f} Cr")
kpi2.metric("Worst-Case (Drought)", f"₹ {worst_case / 10_000_000:,.2f} Cr")
kpi3.metric("Best-Case (Ideal)", f"₹ {best_case / 10_000_000:,.2f} Cr")
kpi4.metric("Systemic Volatility (σ)", f"₹ {volatility / 10_000_000:,.2f} Cr")

# Layout for Timeline and Box Plot side-by-side
col_chart1, col_chart2 = st.columns([2, 1])

with col_chart1:
    # The Timeline Bar Chart
    df_history['Status'] = df_history['Net_Profit'].apply(lambda value: 'Profit' if value >= 0 else 'Deficit')
    fig_timeline = px.bar(
        df_history, x='Year', y='Net_Profit', color='Status',
        color_discrete_map={'Profit': '#27ae60', 'Deficit': '#c0392b'},
        hover_data={'Status': False, 'Year': True, 'Net_Profit': ':,.0f'}
    )
    fig_timeline.add_hline(y=mean_profit, line_dash="dash", line_color="black", annotation_text="Expected Mean")
    fig_timeline.update_layout(xaxis_title="Historical Year", yaxis_title="Portfolio Net Profit (₹)", showlegend=False, height=450)
    st.plotly_chart(fig_timeline, use_container_width=True)

with col_chart2:
    # THE BOX PLOT: Honest, raw data distribution
    fig_box = px.box(
        df_history, 
        y="Net_Profit", 
        points="all",  # This is crucial! It plots the actual 10 historical dots over the box.
        color_discrete_sequence=['#2980b9'],
        hover_data={'Net_Profit': ':,.0f'}
    )
    fig_box.update_layout(
        yaxis_title="", 
        xaxis_title="Risk Distribution", 
        height=450, 
        margin=dict(l=0)
    )
    st.plotly_chart(fig_box, use_container_width=True)