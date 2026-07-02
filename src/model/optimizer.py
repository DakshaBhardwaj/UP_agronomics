import pyomo.environ as pyo
import threading
import numpy as np

# Import the pre-computed data singleton
from src.data.data_loader import db

# Thread lock to prevent race conditions during concurrent API requests 
# mutating the global Pyomo model.
optimization_lock = threading.Lock()

def build_global_model():
    """
    Constructs the Pyomo ConcreteModel globally. 
    Uses Mutable parameters for user inputs so the model doesn't need to be 
    recompiled on every API request, drastically reducing latency.
    """
    print("Compiling Pyomo Model Constraints...")
    m = pyo.ConcreteModel()
    
    # --- Sets ---
    m.D = pyo.Set(initialize=db['regional_districts'])
    m.C = pyo.Set(initialize=db['TARGET_CROPS'])
    m.T = pyo.Set(initialize=[1, 2, 3])
    m.Y = pyo.Set(initialize=db['historical_years'])

    # --- Mutable Parameters (Updated by the API per request) ---
    m.risk_aversion = pyo.Param(mutable=True, default=0.5)
    m.water_cost = pyo.Param(mutable=True, default=25.0)

    # --- Decision Variables ---
    m.Area = pyo.Var(m.D, m.C, domain=pyo.NonNegativeReals)
    m.VolSold = pyo.Var(m.C, m.T, domain=pyo.NonNegativeReals)
    m.HistVolSold = pyo.Var(m.Y, m.C, m.T, domain=pyo.NonNegativeReals)
    
    # Systemic Margin Deviations (For Downside MAD)
    m.DevNeg = pyo.Var(m.Y, domain=pyo.NonNegativeReals) # Downside deviation (Shortfall)
    m.DevPos = pyo.Var(m.Y, domain=pyo.NonNegativeReals) # Upside deviation

    # --- Constraints ---
    
    # 1. Market Demand Capacity (Piecewise Revenue)
    def rule_cap_t1(m, c): return m.VolSold[c, 1] <= float(db['market_demand_tiers'][c][0][0])
    m.Cap_Tier1 = pyo.Constraint(m.C, rule=rule_cap_t1)
    
    def rule_cap_t2(m, c): return m.VolSold[c, 2] <= float(db['market_demand_tiers'][c][1][0])
    m.Cap_Tier2 = pyo.Constraint(m.C, rule=rule_cap_t2)
    
    def rule_sales_link(m, c):
        return sum(m.Area[d, c] * db['expected_yield_matrix'][d][c] for d in m.D) - sum(m.VolSold[c, t] for t in m.T) == 0
    m.Sales_Link = pyo.Constraint(m.C, rule=rule_sales_link)
    
    # Historical Scenarios (For calculating backtest margins)
    def rule_hist_cap_t1(m, y, c): return m.HistVolSold[y, c, 1] <= float(db['market_demand_tiers'][c][0][0])
    m.Hist_Cap_Tier1 = pyo.Constraint(m.Y, m.C, rule=rule_hist_cap_t1)
    
    def rule_hist_cap_t2(m, y, c): return m.HistVolSold[y, c, 2] <= float(db['market_demand_tiers'][c][1][0])
    m.Hist_Cap_Tier2 = pyo.Constraint(m.Y, m.C, rule=rule_hist_cap_t2)
    
    def rule_hist_sales_link(m, y, c):
        return sum(m.Area[d, c] * db['historical_yield_matrix'][d][c][y] for d in m.D) - sum(m.HistVolSold[y, c, t] for t in m.T) == 0
    m.Hist_Sales_Link = pyo.Constraint(m.Y, m.C, rule=rule_hist_sales_link)

    # 2. Agronomic Constraints
    def rule_land_limit(m, d): return sum(m.Area[d, c] for c in m.C) <= db['arable_land_limits'][d]
    m.Land_Limit = pyo.Constraint(m.D, rule=rule_land_limit)
    
    def rule_mono_crop(m, d, c): return m.Area[d, c] <= 0.35 * db['arable_land_limits'][d]
    m.Mono_Crop_Limit = pyo.Constraint(m.D, m.C, rule=rule_mono_crop)
    
    def rule_rotation(m, d):
        legumes_oilseeds = ['CHICKPEA', 'MUSTARD', 'SOYBEAN', 'GROUNDNUT']
        return sum(m.Area[d, c] for c in legumes_oilseeds) - (0.25 * sum(m.Area[d, c] for c in m.C)) >= 0
    m.Rotation = pyo.Constraint(m.D, rule=rule_rotation)

    # 3. Risk Calculation (Downside Semi-MAD)
    def calculate_systemic_deviation(m, y):
        # Expected systemic margin
        expected_rev = sum(m.VolSold[c, t] * db['market_demand_tiers'][c][t-1][1] for c in m.C for t in m.T)
        expected_water_pen = sum(m.Area[d, c] * 1000.0 * max(0.0, db['hydrological_demand'][c] - db['state_baseline_rainfall']) * m.water_cost for d in m.D for c in m.C)
        expected_margin = expected_rev - expected_water_pen
        
        # Actual margin in historical year y
        actual_rev = sum(m.HistVolSold[y, c, t] * db['market_demand_tiers'][c][t-1][1] for c in m.C for t in m.T)
        actual_water_pen = sum(m.Area[d, c] * 1000.0 * max(0.0, db['hydrological_demand'][c] - db['historical_rainfall'][d][y]) * m.water_cost for d in m.D for c in m.C)
        actual_margin = actual_rev - actual_water_pen
        
        # Deviation Constraint: Actual - Expected - Pos + Neg = 0
        return actual_margin - expected_margin - m.DevPos[y] + m.DevNeg[y] == 0
    m.Deviation_Constraint = pyo.Constraint(m.Y, rule=calculate_systemic_deviation)

    # 4. Objective Function (Maximize Risk-Adjusted Margin)
    def maximize_utility(m):
        total_revenue = sum(m.VolSold[c, t] * db['market_demand_tiers'][c][t-1][1] for c in m.C for t in m.T)
        total_cost = sum(m.Area[d, c] * db['expected_yield_matrix'][d][c] * db['crop_economics'][c]['cost'] for d in m.D for c in m.C)
        water_penalty = sum(m.Area[d, c] * 1000.0 * max(0.0, db['hydrological_demand'][c] - db['state_baseline_rainfall']) * m.water_cost for d in m.D for c in m.C)
        
        # Normalize the Downside MAD penalty. 
        # m.risk_aversion is [0, 1]. We scale the raw deviation so it has a meaningful impact.
        # A scalar multiplier of 0.1 ensures the slider provides a smooth transition.
        expected_downside = sum(m.DevNeg[y] for y in m.Y) / len(db['historical_years'])
        systemic_risk_penalty = expected_downside * m.risk_aversion * 0.1
        
        return total_revenue - total_cost - water_penalty - systemic_risk_penalty
        
    m.Objective = pyo.Objective(rule=maximize_utility, sense=pyo.maximize)

    return m

# Instantiate the global model
global_model = build_global_model()
solver = pyo.SolverFactory('glpk')

def run_optimization(risk_aversion: float, water_cost: float):
    """
    Thread-safe optimization runner.
    Updates mutable parameters and resolves the model.
    """
    with optimization_lock:
        # Update mutable parameters directly (O(1) time complexity)
        global_model.risk_aversion = risk_aversion
        global_model.water_cost = water_cost
        
        results = solver.solve(global_model)
        
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise RuntimeError("GLPK Solver failed. No optimal solution found.")
            
        # Extract Results
        optimal_portfolio = {
            c: float(sum(pyo.value(global_model.Area[d, c]) for d in db['regional_districts'])) 
            for c in db['TARGET_CROPS']
        }
        
        portfolio_history = {}
        for y in db['historical_years']:
            rev_y = 0.0
            for c in db['TARGET_CROPS']:
                vol_y = sum(pyo.value(global_model.Area[d, c]) * db['historical_yield_matrix'][d][c][y] for d in db['regional_districts'])
                rem_vol = vol_y
                for t in [1, 2, 3]:
                    tier_max = db['market_demand_tiers'][c][t-1][0] if t < 3 else 9999999
                    alloc_t = min(rem_vol, tier_max)
                    rev_y += alloc_t * db['market_demand_tiers'][c][t-1][1]
                    rem_vol -= alloc_t
                    
            cost_y = sum(pyo.value(global_model.Area[d, c]) * db['expected_yield_matrix'][d][c] * db['crop_economics'][c]['cost'] for d in db['regional_districts'] for c in db['TARGET_CROPS'])
            water_y = sum(pyo.value(global_model.Area[d, c]) * 1000.0 * max(0.0, db['hydrological_demand'][c] - db['historical_rainfall'][d][y]) * water_cost for d in db['regional_districts'] for c in db['TARGET_CROPS'])
            
            portfolio_history[str(y)] = float(rev_y - cost_y - water_y)

        return {
            'optimal_portfolio': optimal_portfolio,
            'portfolio_history': portfolio_history,
            'status_quo_portfolio': db['status_quo_portfolio'],
            'status': 'success'
        }
