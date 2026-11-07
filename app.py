#!/usr/bin/env python3
"""
FastAPI Web Application: Serves the Live Kinase Dashboard on Render.com
"""

import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd

app = FastAPI(
    title="Kinase Bioactivity & Selectivity Dashboard",
    description="Interactive prioritization tool for EGFR kinase inhibitors",
    version="1.0"
)

HTML_FILE = os.path.join(os.path.dirname(__file__), "docs", "index.html")
LEADS_FILE = os.path.join(os.path.dirname(__file__), "results", "consortium_prioritized_leads.csv")

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    """Serves the interactive Plotly compound prioritization dashboard."""
    if os.path.exists(HTML_FILE):
        with open(HTML_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Dashboard file not found. Please run run_pipeline.py first.</h1>"

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "kinase-affinity-dl"}

@app.get("/api/prioritized-leads")
def get_leads(limit: int = 15):
    """API endpoint returning top consortium prioritized leads in JSON."""
    if os.path.exists(LEADS_FILE):
        df = pd.read_csv(LEADS_FILE)
        return df.head(limit).to_dict(orient="records")
    return {"error": "Leads file not generated yet."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
