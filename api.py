from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pyomo.environ as pyo

app = FastAPI(title="UP Agri-Economics Optimization Engine")

# ==========================================
# 1. PRE-LOAD STATIC DATA (Data is Global, Math is Local)
# ==========================================
print("Loading Database into RAM...")
np.random.seed(42)

TARGET_CROPS = ["RICE", "WHEAT", "SUGARCANE", "MAIZE", "SORGHUM", "CHICKPEA", "MUSTARD", "SOYBEAN", "GROUNDNUT"]

df = pd.read_csv('rain-agriculture.csv')
df_state = df[df['State Name'].str.lower() == 'uttar pradesh']
latest_year = int(df_state['Year'].max())
historical_years = list(range(latest_year - 9, latest_year + 1))
df_decade = df_state[df_state['Year'].isin(historical_years)]

regional_districts = df_decade['Dist Code'].unique().tolist()
expected_yield_matrix = {d: {} for d in regional_districts}
historical_yield_matrix = {d: {c: {} for c in TARGET_CROPS} for d in regional_districts}
historical_rainfall = {d: {} for d in regional_districts}
arable_land_limits = {}
baseline_rainfall = 656.0 

structural_baselines = {
    'WHEAT': {'mean': 3.5}, 'SUGARCANE': {'mean': 80.0}, 'MUSTARD': {'mean': 1.5},
    'SOYBEAN': {'mean': 1.2}, 'GROUNDNUT': {'mean': 1.8}
}

for d in regional_districts:
    df_dist = df_decade[df_decade['Dist Code'] == d]
    area_cols = [f'{c} AREA (1000 ha)' for c in TARGET_CROPS if f'{c} AREA (1000 ha)' in df_dist.columns]
    df_area = df_dist[area_cols].apply(pd.to_numeric, errors='coerce')
    total_land = df_area.sum(axis=1)
    
    arable_land_limits[d] = float(total_land.max()) if not total_land.empty else 100.0
    if arable_land_limits[d] <= 0: arable_land_limits[d] = 100.0 
    
    for c in TARGET_CROPS:
        col_name = f'{c} YIELD (Kg per ha)'
        if col_name in df_dist.columns:
            col_data = pd.to_numeric(df_dist[col_name], errors='coerce')
            mean_y = col_data.mean() / 1000.0
            mean_y = float(mean_y) if pd.notna(mean_y) else structural_baselines.get(c, {}).get('mean', 0.0)
            expected_yield_matrix[d][c] = mean_y
            
            for y in historical_years:
                row = df_dist[df_dist['Year'] == y]
                if row.empty:
                    historical_yield_matrix[d][c][y] = max(0.0, float(np.random.triangular(mean_y*0.20, mean_y, mean_y*1.15))) if mean_y > 0 else 0.0
                    continue
                val = row[col_name].iloc[0]
                try:
                    val = float(val)
                    if np.isnan(val) or val <= 0: raise ValueError
                    historical_yield_matrix[d][c][y] = val / 1000.0
                except (ValueError, TypeError):
                    historical_yield_matrix[d][c][y] = max(0.0, float(np.random.triangular(mean_y*0.20, mean_y, mean_y*1.15))) if mean_y > 0 else 0.0
        else:
            expected_yield_matrix[d][c] = structural_baselines.get(c, {}).get('mean', 0.0)

for d in regional_districts:
    bellwether_mean = expected_yield_matrix[d].get('WHEAT', 3.5)
    for y in historical_years:
        bellwether_actual = historical_yield_matrix[d].get('WHEAT', {}).get(y, bellwether_mean)
        climate_index = bellwether_actual / bellwether_mean if bellwether_mean > 0 else 1.0
        historical_rainfall[d][y] = float(baseline_rainfall * climate_index)
        for c in TARGET_CROPS:
            if f'{c} YIELD (Kg per ha)' not in df_decade.columns:
                correlated_yield = expected_yield_matrix[d][c] * climate_index
                noise = np.random.triangular(-0.05 * correlated_yield, 0, 0.05 * correlated_yield) if correlated_yield > 0 else 0.0
                historical_yield_matrix[d][c][y] = max(0.0, float(correlated_yield + noise))

crop_economics = {
    'RICE': {'price': 21830.0, 'cost': 15000.0}, 'WHEAT': {'price': 22750.0, 'cost': 16000.0},
    'SUGARCANE': {'price': 3150.0, 'cost': 2000.0}, 'MAIZE': {'price': 20900.0, 'cost': 14000.0},
    'SORGHUM': {'price': 31800.0, 'cost': 20000.0}, 'CHICKPEA': {'price': 53330.0, 'cost': 25000.0},
    'MUSTARD': {'price': 56500.0, "cost": 25000.0}, 'SOYBEAN': {'price': 46000.0, 'cost': 22000.0},
    'GROUNDNUT': {'price': 63000.0, 'cost': 35000.0}
}

hydrological_demand = {
    'RICE': 1200, 'WHEAT': 650, 'SUGARCANE': 2000, 'MAIZE': 600, 'SORGHUM': 450, 
    'CHICKPEA': 350, 'MUSTARD': 300, 'SOYBEAN': 500, 'GROUNDNUT': 500
}

base_caps = {
    'RICE': 8000, 'WHEAT': 12000, 'SUGARCANE': 100000, 'MAIZE': 5000, 
    'SORGHUM': 2000, 'CHICKPEA': 4000, 'MUSTARD': 2000, 'SOYBEAN': 3000, 'GROUNDNUT': 4000
}

market_demand_tiers = {}
for c in TARGET_CROPS:
    p = crop_economics[c]['price']
    cost = crop_economics[c]['cost']
    market_demand_tiers[c] = [(base_caps[c], p), (base_caps[c] * 2.5, p * 0.75), (9999999, max(p * 0.35, cost * 0.95))]

print("Data Pipeline Ready. Awaiting API Requests...")

# ==========================================
# 2. THE DYNAMIC API ENDPOINT
# ==========================================
class OptimizationPayload(BaseModel):
    risk_aversion: float
    water_cost: float

@app.post("/optimize")
def run_optimization(payload: OptimizationPayload):
    # THE FIX: We build the Pyomo model INSIDE the route. 
    # This guarantees thread-safety and hard-codes the user's sliders into the math.
    
    model = pyo.ConcreteModel()
    model.D = pyo.Set(initialize=regional_districts)
    model.C = pyo.Set(initialize=TARGET_CROPS)
    model.T = pyo.Set(initialize=[1, 2, 3])
    model.Y = pyo.Set(initialize=historical_years)

    model.Area = pyo.Var(model.D, model.C, domain=pyo.NonNegativeReals)
    model.VolSold = pyo.Var(model.C, model.T, domain=pyo.NonNegativeReals)
    model.HistVolSold = pyo.Var(model.Y, model.C, model.T, domain=pyo.NonNegativeReals)
    model.DevNeg = pyo.Var(model.Y, domain=pyo.NonNegativeReals)
    model.DevPos = pyo.Var(model.Y, domain=pyo.NonNegativeReals)

    # Constraints
    model.Cap_Tier1 = pyo.Constraint(model.C, rule=lambda m, c: m.VolSold[c, 1] <= float(market_demand_tiers[c][0][0]))
    model.Cap_Tier2 = pyo.Constraint(model.C, rule=lambda m, c: m.VolSold[c, 2] <= float(market_demand_tiers[c][1][0]))
    model.Sales_Link = pyo.Constraint(model.C, rule=lambda m, c: sum(m.Area[d, c] * expected_yield_matrix[d][c] for d in m.D) - sum(m.VolSold[c, t] for t in m.T) == 0)
    
    model.Hist_Cap_Tier1 = pyo.Constraint(model.Y, model.C, rule=lambda m, y, c: m.HistVolSold[y, c, 1] <= float(market_demand_tiers[c][0][0]))
    model.Hist_Cap_Tier2 = pyo.Constraint(model.Y, model.C, rule=lambda m, y, c: m.HistVolSold[y, c, 2] <= float(market_demand_tiers[c][1][0]))
    model.Hist_Sales_Link = pyo.Constraint(model.Y, model.C, rule=lambda m, y, c: sum(m.Area[d, c] * historical_yield_matrix[d][c][y] for d in m.D) - sum(m.HistVolSold[y, c, t] for t in m.T) == 0)

    model.Land_Limit = pyo.Constraint(model.D, rule=lambda m, d: sum(m.Area[d, c] for c in m.C) <= arable_land_limits[d])
    model.Mono_Crop_Limit = pyo.Constraint(model.D, model.C, rule=lambda m, d, c: m.Area[d, c] <= 0.35 * arable_land_limits[d])
    model.Rotation = pyo.Constraint(model.D, rule=lambda m, d: sum(m.Area[d, c] for c in ['CHICKPEA', 'MUSTARD', 'SOYBEAN', 'GROUNDNUT']) - (0.25 * sum(m.Area[d, c] for c in m.C)) >= 0)

    def calculate_systemic_deviation(m, y):
        expected_rev = sum(m.VolSold[c, t] * market_demand_tiers[c][t-1][1] for c in TARGET_CROPS for t in [1, 2, 3])
        # Integrating the user's custom water penalty payload directly
        expected_water = sum(m.Area[d, c] * 1000.0 * max(0.0, hydrological_demand[c] - baseline_rainfall) * payload.water_cost for d in regional_districts for c in TARGET_CROPS)
        expected_margin = expected_rev - expected_water
        
        actual_rev = sum(m.HistVolSold[y, c, t] * market_demand_tiers[c][t-1][1] for c in TARGET_CROPS for t in [1, 2, 3])
        actual_water = sum(m.Area[d, c] * 1000.0 * max(0.0, hydrological_demand[c] - historical_rainfall[d][y]) * payload.water_cost for d in regional_districts for c in TARGET_CROPS)
        actual_margin = actual_rev - actual_water
        
        return actual_margin - expected_margin - m.DevPos[y] + m.DevNeg[y] == 0
    model.Deviation_Constraint = pyo.Constraint(model.Y, rule=calculate_systemic_deviation)

    def maximize_mad_adjusted_elastic_utility(m):
        total_revenue = sum(m.VolSold[c, t] * market_demand_tiers[c][t-1][1] for c in TARGET_CROPS for t in [1, 2, 3])
        total_cost = sum(m.Area[d, c] * expected_yield_matrix[d][c] * crop_economics[c]['cost'] for d in regional_districts for c in TARGET_CROPS)
        water_penalty = sum(m.Area[d, c] * 1000.0 * max(0.0, hydrological_demand[c] - baseline_rainfall) * payload.water_cost for d in regional_districts for c in TARGET_CROPS)
        
        # Integrating the user's custom risk aversion payload directly
        systemic_risk_penalty = (sum(m.DevNeg[y] for y in m.Y) / len(historical_years)) * payload.risk_aversion
        return total_revenue - total_cost - water_penalty - systemic_risk_penalty
    model.Objective = pyo.Objective(rule=maximize_mad_adjusted_elastic_utility, sense=pyo.maximize)

    solver = pyo.SolverFactory('glpk')
    results = solver.solve(model)

    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        optimal_portfolio = {c: float(sum(pyo.value(model.Area[d, c]) for d in regional_districts)) for c in TARGET_CROPS} # type: ignore
        
        portfolio_history = {}
        for y in historical_years:
            rev_y = 0.0
            for c in TARGET_CROPS:
                vol_y = sum(pyo.value(model.Area[d, c]) * historical_yield_matrix[d][c][y] for d in regional_districts)
                rem_vol = vol_y
                for t in [1, 2, 3]:
                    tier_max = market_demand_tiers[c][t-1][0] if t < 3 else 9999999
                    alloc_t = min(rem_vol, tier_max)
                    rev_y += alloc_t * market_demand_tiers[c][t-1][1]
                    rem_vol -= alloc_t
                    
            cost_y = sum(pyo.value(model.Area[d, c]) * expected_yield_matrix[d][c] * crop_economics[c]['cost'] for d in regional_districts for c in TARGET_CROPS)
            water_y = sum(pyo.value(model.Area[d, c]) * 1000.0 * max(0.0, hydrological_demand[c] - historical_rainfall[d][y]) * payload.water_cost for d in regional_districts for c in TARGET_CROPS) # type: ignore
            portfolio_history[str(y)] = float(rev_y - cost_y - water_y)

        return JSONResponse(content={
            'optimal_portfolio': optimal_portfolio,
            'portfolio_history': portfolio_history,
            'status': 'success'
        })
    else:
        raise HTTPException(status_code=500, detail="GLPK Solver failed. No optimal solution found.")