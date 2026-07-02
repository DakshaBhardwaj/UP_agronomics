#  Agri-Economics Optimizer: Climate Risk & Portfolio Optimization

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)
![Pyomo](https://img.shields.io/badge/Pyomo-Optimization-orange.svg)

An enterprise-grade **Operations Research** and **Data Science** engine designed to mitigate systemic climate risk in subsistence agriculture. 

Traditional agricultural models focus on maximizing expected yield, which often pushes farmers to plant highly profitable but water-sensitive crops. When a monsoon fails (drought), these portfolios suffer catastrophic financial ruin. This engine solves that problem by using **Linear Programming (GLPK)** and **Downside Risk (Semi-MAD)** to generate a mathematically optimal crop portfolio that maximizes revenue while strictly bounding worst-case financial losses.

---

##  The Mathematical Engine

This project bridges empirical data science with convex optimization.

### 1. Empirical Covariate Imputation (Yield Estimation)
To measure risk accurately, we must preserve the true covariance of crop failures. Synthetic random noise cannot simulate a drought. Instead, we extract true historical monsoon rainfall ($R_y$) anomalies to estimate historical yields ($Y_{c,y}$) using a drought-resistance coefficient ($\alpha_c$):

$$ Y_{c,y} = \mu_c \cdot \left( \alpha_c + (1 - \alpha_c)\frac{R_y}{\bar{R}} \right) + \epsilon $$

*This ensures that in our simulated back-test, a drought penalizes all water-sensitive crops simultaneously, simulating a true systemic shock.*

### 2. Downside Risk Optimization (Semi-MAD)
Standard variance (Markowitz) penalizes *all* deviation from the mean, but farmers only care about downside loss. We implemented Downside Mean Absolute Deviation (Semi-MAD). We calculate the Expected Margin $E[M]$ and track the shortfall ($\delta^-_y$) for every historical scenario $y$:

$$ M_y - E[M] - \delta^+_y + \delta^-_y = 0 \quad \text{(where } \delta^-_y \ge 0 \text{)} $$

The Pyomo **Objective Function** maximizes Expected Revenue minus a dynamically scaled penalty ($\lambda$) for expected shortfalls:

$$ \max \left( E[M] - \text{Water Penalty} - \lambda \cdot E[\delta^-] \right) $$

### 3. Piecewise Concave Demand Constraints
To model market elasticity natively within a Linear Programming (LP) framework and avoid the computational complexity of Mixed-Integer Programming (MIP), we utilize piecewise concave revenue tiers. The model is forced to fulfill Tier 1 (Base Price) before expanding to Tier 2 (Lower Price), naturally preventing infinite mono-cropping.

---

##  Software Architecture

This project features a strictly decoupled frontend/backend architecture designed for ultra-low latency.

- **Backend (FastAPI & Pyomo):** Pyomo optimization matrices are notoriously slow to build. Instead of rebuilding the `ConcreteModel` on every API request, the matrix is instantiated globally in memory at server startup. API requests utilize a `threading.Lock` to safely mutate specific parameters (`pyo.Param(mutable=True)`). 
- **Frontend (Streamlit):** A dynamic dashboard that visualizes the "Status Quo" actual crop allocations versus the "LP Optimized" recommendations.
- **Latency:** By avoiding re-compilation, the API solves a 10-year systemic LP matrix and returns JSON in `< 50ms`.

---

##  Model Evaluation: Pros & Cons

###  Pros (Strengths)
1. **Avoids Bankruptcy:** Directly targets and minimizes catastrophic downside risk, making it highly applicable for crop insurance pricing and government drought-subsidy planning.
2. **Computational Efficiency:** Reduces non-linear market elasticity into a piecewise linear problem, guaranteeing global optimum convergence via GLPK without slow branch-and-bound MIP solvers.
3. **True Covariance:** Escapes the trap of independent Monte Carlo simulations by anchoring all yield variance to a single exogenous empirical variable (monsoon rainfall).

###  Cons (Limitations to Acknowledge)
1. **The Linearity Fallacy:** The imputation formula models yield as a strictly linear function of rainfall. In reality, biological response to water is parabolic (a bell curve). By ignoring the right-side tail, the model accounts for droughts but ignores the crop destruction caused by excessive monsoon flooding.
2. **Static Price Elasticity:** While the model uses demand tiers, the base prices are static across scenarios. If a massive drought occurs, crop supply crashes, which *should* cause market prices to spike. The model ignores this inverse price elasticity, causing it to theoretically overestimate the financial damage of a drought.
3. **Spatial Homogeneity:** The model aggregates land constraints (e.g., Uttar Pradesh) into a singular mega-farm, ignoring localized logistics, transportation costs, and micro-soil environments (e.g., Arid Bundelkhand vs. the fertile Doab).

---

##  Installation & Usage

### 1. Requirements
Ensure you have Python 3.10+ and the GLPK solver installed on your system.
* **Windows:** `winget install glpk`
* **Linux (Ubuntu):** `sudo apt-get install glpk-utils`
* **Mac (Homebrew):** `brew install glpk`

### 2. Setup
Clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/agri-optimizer.git
cd agri-optimizer
pip install -r requirements.txt
```

### 3. Running the Stack
You need to run both the FastAPI backend and the Streamlit frontend.

**Terminal 1 (Backend):**
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```bash
streamlit run src/dashboard/app.py
```

Navigate to `http://localhost:8501` to interact with the dashboard.

---
*Built as an exploration of Operations Research, Linear Programming, and robust architectural design.*
