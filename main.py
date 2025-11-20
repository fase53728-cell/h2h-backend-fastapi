from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from models import League, Team, H2HAnalysis
from services.h2h_analyzer import list_leagues, list_teams, analyze_h2h

app = FastAPI(
    title="H2H Predictor Backend",
    version="1.0.0",
    description="Backend FastAPI para análise H2H usando CSVs por time organizados por liga.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["status"])
def root():
    return {
        "message": "H2H Predictor API funcionando",
        "status": "online",
    }


@app.get("/leagues", response_model=List[League], tags=["ligas"])
def get_leagues():
    return list_leagues()


@app.get("/league/{league_id}/teams", response_model=List[Team], tags=["times"])
def get_teams(league_id: str):
    teams = list_teams(league_id)
    if not teams:
        raise HTTPException(status_code=404, detail="Liga não encontrada ou sem times")
    return teams


@app.get("/league/{league_id}/h2h", response_model=H2HAnalysis, tags=["h2h"])
def get_h2h(
    league_id: str,
    home: str = Query(..., description="Nome do time mandante (conforme nome do CSV)"),
    away: str = Query(..., description="Nome do time visitante (conforme nome do CSV)"),
):
    try:
        return analyze_h2h(league_id, home, away)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {e}")
