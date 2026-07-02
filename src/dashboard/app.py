import streamlit as st
import pandas as pd
import plotly.express as px
import requests

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="UP Agri Portfolio Optimizer", layout="wide", page_icon="🌾")
st.title("🌾 Uttar Pradesh Crop Portfolio Optimizer")
st.markdown("""
**Enterprise-Grade Risk-Adjusted Agricultural Planning**  
This engine utilizes a Linear Programming (LP) model with a **Downside Mean Absolute Deviation (Semi-MAD)** penalty to optimize crop portfolios. 
Unlike standard models, it simulates historical scenarios using **true empirical covariate imputation** (actual monsoon rainfall) to accurately model systemic climate risks like drought.
""")

# --- INTERACTIVE SIDEBAR WITH FORM BATCHING ---
st.sidebar.header("⚙️ Optimization Parameters")
st.sidebar.markdown("Configure the objective parameters and resolve the Pyomo matrix.")

with st.sidebar.form(key='optimization_form'):
    user_risk = st.slider(
        "Risk Aversion Penalty (λ)", 
        min_value=0.0, max_value=1.0, value=0.50, step=0.05, 
        help="0.0 = Maximize pure expected margin. 1.0 = Maximize safety against systemic downside risks (droughts)."
    )

    user_water_cost = st.slider(
        "Groundwater Pumping Cost (₹/mm)", 
        min_value=10.0, max_value=50.0, value=25.0, step=1.0,
        help="Increases the financial penalty for planting water-heavy crops like Sugarcane in districts with historically low rainfall."
    )
    
    submit_button = st.form_submit_button(label="🚀 Run LP Simulation")

st.sidebar.divider()
st.sidebar.info("""
**Architecture Note:**  
The Pyomo LP model is held in memory as a global mutable state in the FastAPI backend, 
meaning slider adjustments solve the matrix in under 50ms without recompiling the structural constraints.
""")

API_URL = "http://127.0.0.1:8000/optimize"

# ==========================================
# 2. API CONNECTION & DATA FETCHING
# ==========================================
@st.cache_data(ttl=3600)
def fetch_portfolio_state(risk, water):
    """Fetches the optimal portfolio from the Pyomo engine via REST API."""
    payload = {"risk_aversion": risk, "water_cost": water}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Cannot connect to the backend Optimization Engine. Ensure the FastAPI server is running (`python -m src.api.main`). Error: {e}")
        st.stop()

if submit_button:
    st.cache_data.clear()

with st.spinner("Injecting mutable constraints & solving Pyomo matrix..."):
    state = fetch_portfolio_state(user_risk, user_water_cost)

# ==========================================
# 3. DASHBOARD VISUALIZATIONS
# ==========================================

# --- SECTION A: Optimal Allocation & Status Quo Comparison ---
st.subheader("📊 1. Arable Land Allocation: Status Quo vs. Optimized")
st.caption("Comparing what farmers currently plant (latest historical year) vs. the LP Recommendation.")

# Prepare comparison dataframe
status_quo = state.get('status_quo_portfolio', {})
optimal = state['optimal_portfolio']

comp_data = []
for c in optimal.keys():
    # Only include crops that have >0 acreage in either scenario
    if optimal[c] > 0 or status_quo.get(c, 0) > 0:
        comp_data.append({'Crop': c, 'Hectares': status_quo.get(c, 0) * 1000, 'Scenario': 'Current (Status Quo)'})
        comp_data.append({'Crop': c, 'Hectares': optimal[c] * 1000, 'Scenario': 'LP Optimized'})

df_comp = pd.DataFrame(comp_data)

if df_comp.empty:
    st.warning("The solver returned an empty portfolio. Try adjusting the parameters to be less strict.")
else:
    col_alloc1, col_alloc2 = st.columns([2, 1])
    
    with col_alloc1:
        # Grouped Bar Chart for Comparison
        fig_bar = px.bar(
            df_comp, x='Crop', y='Hectares', color='Scenario', barmode='group',
            color_discrete_map={'Current (Status Quo)': '#95a5a6', 'LP Optimized': '#2980b9'},
            title="Acreage Comparison"
        )
        fig_bar.update_layout(yaxis_title="Allocated Land (Hectares)", xaxis_title="")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_alloc2:
        # Treemap for Optimal Only
        df_optimal_only = df_comp[df_comp['Scenario'] == 'LP Optimized']
        fig_tree = px.treemap(
            df_optimal_only, 
            path=[px.Constant("LP Optimized"), 'Crop'], 
            values='Hectares',
            color='Crop',
            color_discrete_sequence=px.colors.qualitative.Prism,
            title="Optimized Composition"
        )
        fig_tree.update_traces(textinfo="label+percent parent")
        fig_tree.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=350)
        st.plotly_chart(fig_tree, use_container_width=True)

st.divider()

# --- SECTION B: Historical Back-Test ---
st.subheader("📉 2. Empirical Downside Risk Back-Test")
st.caption("A deterministic replay over the last 10 historical years, using true rainfall covariates to calculate systemic deviations.")

history_data = [{'Year': str(k), 'Net_Profit': v} for k, v in state['portfolio_history'].items()]
df_history = pd.DataFrame(history_data).sort_values("Year")

mean_profit = df_history['Net_Profit'].mean()
worst_case = df_history['Net_Profit'].min()
best_case = df_history['Net_Profit'].max()
volatility = df_history['Net_Profit'].std()

# KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Expected Mean Margin", f"₹ {mean_profit / 10_000_000:,.2f} Cr")
kpi2.metric("Worst-Case (Downside)", f"₹ {worst_case / 10_000_000:,.2f} Cr")
kpi3.metric("Best-Case (Upside)", f"₹ {best_case / 10_000_000:,.2f} Cr")
kpi4.metric("Systemic Volatility (σ)", f"₹ {volatility / 10_000_000:,.2f} Cr")

col_chart1, col_chart2 = st.columns([2, 1])

with col_chart1:
    df_history['Status'] = df_history['Net_Profit'].apply(lambda value: 'Surplus' if value >= mean_profit else 'Shortfall')
    fig_timeline = px.bar(
        df_history, x='Year', y='Net_Profit', color='Status',
        color_discrete_map={'Surplus': '#27ae60', 'Shortfall': '#e74c3c'},
        hover_data={'Status': False, 'Year': True, 'Net_Profit': ':,.0f'}
    )
    fig_timeline.add_hline(y=mean_profit, line_dash="dash", line_color="white", annotation_text="Expected Mean")
    fig_timeline.update_layout(xaxis_title="Historical Year", yaxis_title="Systemic Portfolio Margin (₹)", showlegend=True, height=450)
    st.plotly_chart(fig_timeline, use_container_width=True)

with col_chart2:
    fig_box = px.box(
        df_history, 
        y="Net_Profit", 
        points="all", 
        color_discrete_sequence=['#2980b9'],
        hover_data={'Net_Profit': ':,.0f'}
    )
    fig_box.update_layout(yaxis_title="", xaxis_title="Margin Distribution", height=450, margin=dict(l=0))
    st.plotly_chart(fig_box, use_container_width=True)
