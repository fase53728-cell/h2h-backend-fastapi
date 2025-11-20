from pathlib import Path
from typing import List
import pandas as pd

from config import DATA_DIR
from utils.team_normalizer import normalize_name
from models import League, Team, TeamStats, H2HAnalysis, H2HPrediction


def list_leagues() -> List[League]:
    leagues: List[League] = []
    if not DATA_DIR.exists():
        return leagues

    for item in DATA_DIR.iterdir():
        if item.is_dir():
            league_id = item.name
            name = item.name.replace("-", " ").title()
            leagues.append(League(league_id=league_id, name=name))
    return leagues


def list_teams(league_id: str) -> List[Team]:
    league_path = DATA_DIR / league_id
    teams: List[Team] = []

    if not league_path.exists() or not league_path.is_dir():
        return teams

    for csv_path in league_path.glob("*.csv"):
        display_name = csv_path.stem
        team_id = normalize_name(display_name)
        teams.append(Team(team_id=team_id, name=display_name))
    return teams


def _find_team_file(league_id: str, team_name: str) -> Path:
    league_path = DATA_DIR / league_id
    if not league_path.exists() or not league_path.is_dir():
        raise FileNotFoundError(f"Liga não encontrada: {league_id}")

    target_norm = normalize_name(team_name)
    for csv_path in league_path.glob("*.csv"):
        stem_norm = normalize_name(csv_path.stem)
        if stem_norm == target_norm:
            return csv_path

    raise FileNotFoundError(f"Time '{team_name}' não encontrado na liga '{league_id}'")


def _load_team_stats(league_id: str, team_name: str) -> TeamStats:
    csv_path = _find_team_file(league_id, team_name)

    # CSV padrão: ; como separador (padrão Angers / seu modelo)
    df = pd.read_csv(csv_path, sep=";", engine="python")

    if df.empty:
        raise ValueError(f"CSV vazio para o time {team_name} ({csv_path})")

    row = df.iloc[0]

    numeric_cols = {}
    for col in df.columns:
        try:
            val = float(row[col])
            numeric_cols[col] = val
        except (ValueError, TypeError):
            continue

    # Aqui mantemos estatísticas "básicas" em branco.
    # Depois você configura o motor PRO usando as colunas certas.
    return TeamStats(
        team_id=normalize_name(team_name),
        name=team_name,
        raw_columns=numeric_cols,
        win_prob=None,
        draw_prob=None,
        loss_prob=None,
        avg_goals_for=None,
        avg_goals_against=None,
        over_15_prob=None,
        over_25_prob=None,
        btts_prob=None,
    )


def analyze_h2h(league_id: str, home_name: str, away_name: str) -> H2HAnalysis:
    home_stats = _load_team_stats(league_id, home_name)
    away_stats = _load_team_stats(league_id, away_name)

    prediction = H2HPrediction(
        best_bet="EM_ANALISE",
        comment=(
            "Motor básico ativo. Configure depois o motor PRO para gerar "
            "prognósticos inteligentes (1, X1, X, X2, 2, tendências de gols, etc.)."
        ),
    )

    return H2HAnalysis(
        league_id=league_id,
        home=home_stats,
        away=away_stats,
        prediction=prediction,
    )
