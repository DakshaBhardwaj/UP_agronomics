import pandas as pd
import numpy as np

# ==========================================
# 1. CONSTANTS AND ECONOMICS DATA
# ==========================================
TARGET_CROPS = ["RICE", "WHEAT", "SUGARCANE", "MAIZE", "SORGHUM", "CHICKPEA", "MUSTARD", "SOYBEAN", "GROUNDNUT"]

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

structural_baselines = {
    'WHEAT': {'mean': 3.5}, 'SUGARCANE': {'mean': 80.0}, 'MUSTARD': {'mean': 1.5},
    'SOYBEAN': {'mean': 1.2}, 'GROUNDNUT': {'mean': 1.8}
}

market_demand_tiers = {}
for c in TARGET_CROPS:
    p = crop_economics[c]['price']
    cost = crop_economics[c]['cost']
    # Tier 1 (Base Cap, Full Price), Tier 2 (2.5x Base Cap, 75% Price), Tier 3 (Infinite, Max(35% Price, 95% Cost))
    market_demand_tiers[c] = [
        (base_caps[c], p), 
        (base_caps[c] * 2.5, p * 0.75), 
        (9999999, max(p * 0.35, cost * 0.95))
    ]

# Drought vulnerability factors (1.0 = Highly Vulnerable/Responsive, 0.0 = Totally Resistant)
drought_vulnerability = {
    'RICE': 0.9, 'SUGARCANE': 0.8, 'MAIZE': 0.6, 'WHEAT': 0.5, 
    'SOYBEAN': 0.4, 'GROUNDNUT': 0.4, 'MUSTARD': 0.3, 'CHICKPEA': 0.2, 'SORGHUM': 0.1
}

# ==========================================
# 2. DATA PROCESSING & IMPUTATION
# ==========================================
def load_and_process_data(filepath='rain-agriculture.csv'):
    """
    Loads historical agriculture data, computes true rainfall covariates, 
    and returns fully imputed matrices for Pyomo LP.
    """
    print("Loading and processing data from CSV...")
    np.random.seed(42)
    
    df = pd.read_csv(filepath)
    df_state = df[df['State Name'].str.lower() == 'uttar pradesh']
    
    if df_state.empty:
        raise ValueError("Uttar Pradesh data not found in CSV.")
        
    latest_year = int(df_state['Year'].max())
    historical_years = list(range(latest_year - 9, latest_year + 1))
    df_decade = df_state[df_state['Year'].isin(historical_years)]
    
    regional_districts = df_decade['Dist Code'].unique().tolist()
    
    expected_yield_matrix = {d: {} for d in regional_districts}
    historical_yield_matrix = {d: {c: {} for c in TARGET_CROPS} for d in regional_districts}
    historical_rainfall = {d: {} for d in regional_districts}
    arable_land_limits = {}
    
    # Calculate baseline average rainfall for the state over the decade
    # Rainfall is the sum of JUN, JUL, AUG, SEP
    df_decade['Total_Rainfall'] = df_decade[['JUN', 'JUL', 'AUG', 'SEP']].sum(axis=1)
    state_baseline_rainfall = float(df_decade['Total_Rainfall'].mean())
    if pd.isna(state_baseline_rainfall) or state_baseline_rainfall == 0:
        state_baseline_rainfall = 656.0 # Fallback
        
    for d in regional_districts:
        df_dist = df_decade[df_decade['Dist Code'] == d]
        
        # 1. Rainfall Processing
        dist_avg_rain = float(df_dist['Total_Rainfall'].mean())
        if pd.isna(dist_avg_rain) or dist_avg_rain == 0:
            dist_avg_rain = state_baseline_rainfall
            
        for y in historical_years:
            row = df_dist[df_dist['Year'] == y]
            if not row.empty and not pd.isna(row['Total_Rainfall'].iloc[0]):
                historical_rainfall[d][y] = float(row['Total_Rainfall'].iloc[0])
            else:
                historical_rainfall[d][y] = dist_avg_rain
        
        # 2. Arable Land Limits
        area_cols = [f'{c} AREA (1000 ha)' for c in TARGET_CROPS if f'{c} AREA (1000 ha)' in df_dist.columns]
        df_area = df_dist[area_cols].apply(pd.to_numeric, errors='coerce')
        total_land = df_area.sum(axis=1)
        
        limit = float(total_land.max()) if not total_land.empty else 100.0
        arable_land_limits[d] = limit if limit > 0 else 100.0 
        
        # 3. Yield Matrix and Imputation
        for c in TARGET_CROPS:
            col_name = f'{c} YIELD (Kg per ha)'
            
            mean_y = structural_baselines.get(c, {}).get('mean', 0.0)
            if col_name in df_dist.columns:
                col_data = pd.to_numeric(df_dist[col_name], errors='coerce')
                calc_mean = col_data.mean() / 1000.0
                if pd.notna(calc_mean) and calc_mean > 0:
                    mean_y = float(calc_mean)
            
            expected_yield_matrix[d][c] = mean_y
            
            vuln = drought_vulnerability.get(c, 0.5)
            
            for y in historical_years:
                row = df_dist[df_dist['Year'] == y]
                val = np.nan
                if not row.empty and col_name in row.columns:
                    val = row[col_name].iloc[0]
                    
                try:
                    val = float(val) / 1000.0
                    if np.isnan(val) or val <= 0: 
                        raise ValueError
                    historical_yield_matrix[d][c][y] = val
                except (ValueError, TypeError):
                    # COVARIATE IMPUTATION: Use rainfall anomaly to estimate yield
                    # Y_est = Mean * (1 - Vulnerability + Vulnerability * (Rain_y / Rain_mean)) + Noise
                    rain_ratio = historical_rainfall[d][y] / dist_avg_rain
                    estimated_yield = mean_y * ((1.0 - vuln) + vuln * rain_ratio)
                    
                    # Add small bound noise (5%) to prevent perfect collinearity
                    noise = np.random.triangular(-0.05 * estimated_yield, 0, 0.05 * estimated_yield) if estimated_yield > 0 else 0.0
                    historical_yield_matrix[d][c][y] = max(0.0, float(estimated_yield + noise))

    # 4. Status Quo Extraction
    # Extract the actual land allocation from the most recent historical year
    df_latest = df_decade[df_decade['Year'] == latest_year]
    status_quo_portfolio = {}
    for c in TARGET_CROPS:
        col_name = f'{c} AREA (1000 ha)'
        if col_name in df_latest.columns:
            val = df_latest[col_name].sum()
            status_quo_portfolio[c] = float(val) if pd.notna(val) else 0.0
        else:
            status_quo_portfolio[c] = 0.0

    print("Data processing complete.")
    return {
        'TARGET_CROPS': TARGET_CROPS,
        'regional_districts': regional_districts,
        'historical_years': historical_years,
        'expected_yield_matrix': expected_yield_matrix,
        'historical_yield_matrix': historical_yield_matrix,
        'historical_rainfall': historical_rainfall,
        'arable_land_limits': arable_land_limits,
        'state_baseline_rainfall': state_baseline_rainfall,
        'crop_economics': crop_economics,
        'hydrological_demand': hydrological_demand,
        'base_caps': base_caps,
        'market_demand_tiers': market_demand_tiers,
        'status_quo_portfolio': status_quo_portfolio
    }

# Execute on import to create a singleton loaded state
db = load_and_process_data()
