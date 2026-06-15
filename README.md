# UP Agronomics — Uttar Pradesh Crop Portfolio Optimizer

> A **risk-adjusted, linear programming model** for optimizing multi-crop agricultural allocation across Uttar Pradesh districts, powered by a decoupled FastAPI optimization engine and an interactive Streamlit dashboard.

---

## Overview

Agriculture in Uttar Pradesh is characterized by high seasonal variability, monsoon dependency, and complex tradeoffs between high-yield cash crops and drought-resilient staples. This project addresses a core question in agro-economics:

> *Given real historical yield data, market price tiers, and water constraints — what is the **optimal hectare allocation** across crops and districts to maximize risk-adjusted profit?*

The model uses **Pyomo** to formulate a **Mean Absolute Deviation (MAD) Linear Programming problem**, solved by the open-source **GLPK solver**. A **FastAPI** backend exposes the solver as a REST API, and a **Streamlit** dashboard lets users interactively control risk aversion and water cost parameters, then visualize the optimized crop portfolio.

---

## Architecture

```
┌──────────────────────────────────┐        POST /optimize
│   Streamlit Dashboard            │ ─────────────────────────►  ┌─────────────────────────────┐
│   (agri_dashboard_final.py)      │                             │   FastAPI Backend (api.py)  │
│                                  │ ◄─────────────────────────  │                             │
│  • Risk Aversion slider (λ)      │    JSON: portfolio +        │  • Loads rain-agriculture   │
│  • Water Cost slider (₹/mm)      │    historical back-test     │    .csv into RAM on startup │
│  • Treemap: crop allocation      │                             │  • Builds Pyomo LP model    │
│  • Bar chart + Box plot (P&L)    │                             │  • Solves with GLPK         │
│  • 4 KPI metric cards            │                             │  • Returns JSON results     │
└──────────────────────────────────┘                             └─────────────────────────────┘
```

The two components run as **separate processes** and communicate over HTTP on `localhost:8000`. This decoupling means:
- The Streamlit UI remains lag-free — sliders don't trigger reruns until the user clicks "Run Dynamic Simulation".
- The solver runs in isolation, making the backend independently testable and replaceable.

---

## Mathematical Model

### Decision Variables

| Variable | Description |
|---|---|
| `Area[d, c]` | Hectares allocated to crop `c` in district `d` (continuous, ≥ 0) |
| `VolSold[c, t]` | Volume of crop `c` sold in price tier `t` |
| `HistVolSold[y, c, t]` | Historical volume sold in year `y`, crop `c`, tier `t` |
| `DevNeg[y]`, `DevPos[y]` | Downside and upside deviations from expected profit in year `y` |

### Objective Function

The model **maximizes a MAD-adjusted elastic utility**:

```
Maximize:
  - Total Revenue (tiered pricing)
  − Total Crop Production Costs
  − Groundwater Pumping Penalty (user-defined ₹/mm)
  − λ × Mean Downside Deviation (MAD risk penalty)
```

Where **λ (risk aversion)** is a user-controlled parameter:
- `λ = 0.0` → Pure profit maximization (high drought risk)
- `λ = 1.0` → Maximum safety against weather shocks (conservative)

### Constraints

| Constraint | Description |
|---|---|
| **Land Limit** | Total area per district ≤ historical max arable land |
| **Monoculture Cap** | Any single crop ≤ 35% of a district's arable land |
| **Crop Rotation** | Legumes/oilseeds (Chickpea, Mustard, Soybean, Groundnut) must cover ≥ 25% of each district |
| **Tiered Market Caps** | Sales volume per crop bounded by three demand tiers with diminishing prices |
| **Sales Linkage** | Total volume sold = total yield produced (no waste/surplus allowed) |
| **MAD Deviation** | Historical profit deviations from expected profit are tracked year-by-year |

### Tiered Pricing Model

Each crop has three market price tiers to model demand elasticity:

| Tier | Volume Cap | Price |
|---|---|---|
| Tier 1 | Base cap (e.g., 8,000 t for Rice) | Full MSP price |
| Tier 2 | 2.5× base cap | 75% of MSP |
| Tier 3 | Unlimited | 35% of MSP (floor) |

---

## Crops Modelled

| Crop | Water Demand (mm) | Price (₹/t) | Production Cost (₹/t) |
|---|---|---|---|
| Rice | 1,200 | 21,830 | 15,000 |
| Wheat | 650 | 22,750 | 16,000 |
| Sugarcane | 2,000 | 3,150 | 2,000 |
| Maize | 600 | 20,900 | 14,000 |
| Sorghum | 450 | 31,800 | 20,000 |
| Chickpea | 350 | 53,330 | 25,000 |
| Mustard | 300 | 56,500 | 25,000 |
| Soybean | 500 | 46,000 | 22,000 |
| Groundnut | 500 | 63,000 | 35,000 |

Water costs are penalized when crop demand **exceeds** baseline district rainfall (656 mm), multiplied by the user-defined pumping cost parameter. 
These are the current prices of the crops which are subject to market changes. 

---

## Dashboard Features

### 1. Recommended Crop Mix (Treemap)
Displays the optimal land allocation in thousands of hectares. A treemap is used instead of a pie chart for clearer comparison of hierarchical area allocations. Crops with zero allocated area are hidden.

### 2. Historical Net-Profit Back-Test
A deterministic replay of the optimized portfolio over the **last 10 historical years** from the dataset:

- **Bar chart** — year-by-year profit/deficit with an expected mean line overlay
- **Box plot** — raw distribution of all historical profits, revealing true tail risk
- **4 KPI cards** — Expected Mean Profit, Worst-Case (drought year), Best-Case, and Systemic Volatility (σ)

### Sidebar Controls (Form-Batched)

| Parameter | Range | Effect |
|---|---|---|
| **Risk Aversion (λ)** | 0.0 – 1.0 | Higher → shifts allocation toward drought-resilient, low-water crops |
| **Groundwater Pumping Cost (₹/mm)** | ₹10 – ₹50 | Higher → penalizes water-heavy crops like Sugarcane in dry districts |

Controls are wrapped in a Streamlit form so UI adjustments are instant and the Pyomo solver only runs when the **"🚀 Run Dynamic Simulation"** button is clicked.

---

## Repository Structure

```
UP_agronomics/
├── api.py                    # FastAPI optimization engine (Pyomo + GLPK)
├── agri_dashboard_final.py   # Streamlit dashboard (UI + Plotly visualizations)
├── rain-agriculture.csv      # Historical district-level yield and rainfall data
├── requirements.txt          # Python dependencies
├── pyrightconfig.json        # Pyright type checker configuration
└── .gitignore
```

---

## Prerequisites

- **Python 3.10+**
- **GLPK (GNU Linear Programming Kit)** — a system-level binary that Pyomo calls as a subprocess. **This must be installed before running the project.** Pyomo itself is a Python modelling layer; it cannot solve any LP problem without a solver binary like GLPK available on your system `PATH`.

>  **If GLPK is not installed, the API will start but every call to `/optimize` will fail** with a `SolverNotAvailable` or `ApplicationError` from Pyomo.

---

## Installing GLPK (Required — All Platforms)

### Linux (Ubuntu / Debian)

The simplest install — one command, no PATH configuration needed:

```bash
sudo apt-get update
sudo apt-get install -y glpk-utils
```

Verify the install:

```bash
glpsol --version
# Expected output: GLPSOL: GLPK LP/MIP Solver, v5.x
```

For other distros:

```bash
# Fedora / RHEL / CentOS
sudo dnf install glpk-utils

# Arch Linux
sudo pacman -S glpk
```

---

### macOS

**Option A — Homebrew (recommended):**

```bash
brew install glpk
```

Verify:

```bash
glpsol --version
```

**Option B — Conda:**

```bash
conda install -c conda-forge glpk
```

---

### Windows

Windows requires a manual download and PATH setup. Follow these steps carefully:

**Step 1 — Download the binaries**

Go to the official WinGLPK SourceForge page and download the latest zip:

> 🔗 https://sourceforge.net/projects/winglpk/files/winglpk/

Download the file named `winglpk-X.XX.zip` (e.g. `winglpk-4.65.zip`).

**Step 2 — Extract**

Extract the zip to a permanent location, for example:

```
C:\glpk\
```

The extracted folder will contain two subdirectories:
- `w32\` — 32-bit binaries
- `w64\` — 64-bit binaries (use this on modern systems)

**Step 3 — Add to System PATH**

1. Press `Win + S`, search for **"Environment Variables"**, and open **"Edit the system environment variables"**.
2. Click **"Environment Variables..."**.
3. Under **"System variables"**, find `Path` and click **"Edit"**.
4. Click **"New"** and add the path to your `w64` folder:
   ```
   C:\glpk\glpk-4.65\w64
   ```
5. Click **OK** on all dialogs.

**Step 4 — Verify in a new terminal**

Open a **new** Command Prompt or PowerShell window (the old one won't have the updated PATH):

```cmd
glpsol --version
```

Expected output:
```
GLPSOL: GLPK LP/MIP Solver, v4.65
```

**Alternative — Chocolatey (if you have it):**

```powershell
choco install glpk -y
```

---

### All Platforms — Conda Alternative

If you use Anaconda or Miniconda, this is the easiest cross-platform option:

```bash
conda install -c conda-forge glpk
```

This works on Linux, macOS, and Windows without any PATH configuration.

---

### Verify Pyomo Can See GLPK

After installing GLPK, confirm that Pyomo can detect it:

```bash
python -c "from pyomo.environ import SolverFactory; s = SolverFactory('glpk'); print('GLPK available:', s.available())"
```

Expected output:
```
GLPK available: True
```

If you see `False`, GLPK is either not installed or not on your PATH. Re-check the steps above.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/DakshaBhardwaj/UP_agronomics.git
cd UP_agronomics
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI Backend

Open a terminal and run:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

You should see:

```
Loading Database into RAM...
Data Pipeline Ready. Awaiting API Requests...
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 4. Launch the Streamlit Dashboard

In a **second terminal**, run:

```bash
streamlit run agri_dashboard_final.py
```

The dashboard will open in your browser at `http://localhost:8501`.

>  **Both processes must run simultaneously.** The Streamlit frontend calls the FastAPI backend on `http://127.0.0.1:8000/optimize`. If the backend is not running, the dashboard will display a connection error.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.58 | Interactive web dashboard |
| `fastapi` + `uvicorn` | — | REST API server for the LP solver |
| `pyomo` | ≥ 6.10 | LP/MILP modelling framework |
| `numpy` | ≥ 2.4 | Numerical operations, random triangular distribution |
| `pandas` | ≥ 3.0 | CSV data loading and district-level aggregation |
| `plotly` | ≥ 6.8 | Treemap, bar chart, and box plot visualizations |
| `requests` | ≥ 2.34 | HTTP POST from Streamlit to FastAPI |

---

## Data Source

The model uses `rain-agriculture.csv`, which contains district-level data for Uttar Pradesh including:
- **Crop area** (1000 ha) per district per year
- **Crop yield** (kg/ha) per district per year
- Historical records used to compute a 10-year back-test window

Districts missing yield data for certain crops fall back to structural baseline yields from literature. Missing historical years use a correlated triangular random draw based on that district's wheat yield as a climate proxy.

---

## API Reference

### `POST /optimize`

Runs the Pyomo LP model with user-supplied parameters.

**Request Body:**
```json
{
  "risk_aversion": 0.5,
  "water_cost": 25.0
}
```

**Response:**
```json
{
  "optimal_portfolio": {
    "RICE": 1234.5,
    "WHEAT": 5678.9,
    ...
  },
  "portfolio_history": {
    "2014": 9876543210.0,
    "2015": 8765432109.0,
    ...
  },
  "status": "success"
}
```

- `optimal_portfolio` — total hectares allocated per crop (summed across all districts), in thousands of ha
- `portfolio_history` — net profit (₹) per historical year for the optimized allocation

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## Author

**Daksha Bhardwaj** — [GitHub](https://github.com/DakshaBhardwaj)

---

## 📄 License

This project is open source. See the repository for license details.
