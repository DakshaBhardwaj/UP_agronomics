from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# Import the optimization runner
from src.model.optimizer import run_optimization

app = FastAPI(
    title="UP Agri-Economics Optimization Engine API",
    description="Backend optimization engine for agricultural land allocation."
)

class OptimizationPayload(BaseModel):
    risk_aversion: float
    water_cost: float

@app.post("/optimize")
def optimize_portfolio(payload: OptimizationPayload):
    """
    Receives user parameters, updates the global Pyomo model, 
    runs the GLPK solver, and returns the optimal crop mix.
    """
    try:
        results = run_optimization(payload.risk_aversion, payload.water_cost)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
