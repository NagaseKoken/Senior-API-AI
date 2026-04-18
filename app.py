"""
Football Salary Prediction API
===============================
A FastAPI application serving salary range predictions for football players,
powered by machine learning models and LLM-generated explanations.

Start the server:
    uvicorn app:app --reload

API docs:
    http://localhost:8000/docs
"""

import os
import json
import urllib.request
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import pandas as pd
import joblib
from functools import lru_cache
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

FOOTBALL_API_KEY = os.getenv('FOOTBALL_API_KEY', '')

# ============================================================
# Global State (loaded at startup)
# ============================================================

STATE = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATABASE_URL = os.environ['DATABASE_URL']
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
FIGURES_API_DIR = os.path.join(BASE_DIR, 'figures_api')


def get_db_connection():
    """Return a new PostgreSQL connection."""
    return psycopg2.connect(DATABASE_URL)


def load_artifacts():
    """Load all model artifacts into memory."""
    s = {}
    s['rf_model'] = joblib.load(os.path.join(MODELS_DIR, 'rf_model.joblib'))
    s['best_model'] = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
    s['scaler'] = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    s['feature_names'] = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))
    s['le_foot'] = joblib.load(os.path.join(MODELS_DIR, 'label_encoder_foot.joblib'))
    s['feature_medians'] = joblib.load(os.path.join(MODELS_DIR, 'feature_medians.joblib'))
    s['position_categories'] = joblib.load(os.path.join(MODELS_DIR, 'position_categories.joblib'))
    s['players_df'] = joblib.load(os.path.join(MODELS_DIR, 'all_predictions.joblib'))
    s['feature_matrix'] = joblib.load(os.path.join(MODELS_DIR, 'feature_matrix.joblib'))

    with open(os.path.join(MODELS_DIR, 'model_config.json')) as f:
        s['model_config'] = json.load(f)
    with open(os.path.join(MODELS_DIR, 'model_metrics.json')) as f:
        s['model_metrics'] = json.load(f)
    with open(os.path.join(MODELS_DIR, 'feature_importances.json')) as f:
        s['feature_importances'] = json.load(f)
    with open(os.path.join(MODELS_DIR, 'dataset_stats.json')) as f:
        s['dataset_stats'] = json.load(f)

    # Pre-compute scaled feature matrix for similarity search
    s['scaled_matrix'] = s['scaler'].transform(s['feature_matrix'])

    return s


def enrich_players_df_from_db():
    """Add api_football_id, nationality (from player_identity) and club_name
    (from sofifa_attributes) to players_df. Missing values become NaN.
    """
    df = STATE['players_df']
    df['api_football_id'] = pd.NA
    df['nationality'] = pd.NA
    df['club_name'] = pd.NA
    try:
        conn = get_db_connection()
        try:
            pks = [int(x) for x in df['player_pk'].tolist()]
            cur = conn.cursor()
            cur.execute(
                "SELECT player_pk, api_football_id, nationality FROM player_identity "
                "WHERE player_pk = ANY(%s)",
                (pks,),
            )
            identity = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
            cur.execute(
                "SELECT player_pk, club_name FROM sofifa_attributes "
                "WHERE player_pk = ANY(%s)",
                (pks,),
            )
            clubs = {r[0]: r[1] for r in cur.fetchall()}
            cur.close()
        finally:
            conn.close()

        df['api_football_id'] = df['player_pk'].map(lambda pk: identity.get(int(pk), (None, None))[0])
        df['nationality'] = df['player_pk'].map(lambda pk: identity.get(int(pk), (None, None))[1])
        df['club_name'] = df['player_pk'].map(lambda pk: clubs.get(int(pk)))
        print(f"   Enriched {df['api_football_id'].notna().sum()} players with api_football_id, "
              f"{df['nationality'].notna().sum()} with nationality, "
              f"{df['club_name'].notna().sum()} with club_name")
    except Exception as e:
        print(f"   WARNING: Could not enrich players_df from DB: {e}")


UNTRAINED_LEAGUE_ALIASES = {
    'Pro League': 'Saudi Pro League',
}


def _map_sofifa_position(player_positions):
    if not player_positions:
        return None
    first = str(player_positions).split(',')[0].strip().upper()
    if first == 'GK':
        return 'Goalkeeper'
    if first in ('CB', 'LB', 'RB', 'LWB', 'RWB'):
        return 'Defender'
    if first in ('CDM', 'CM', 'CAM', 'LM', 'RM'):
        return 'Midfielder'
    if first in ('LW', 'RW', 'CF', 'ST'):
        return 'Attacker'
    return None


def load_untrained_players_from_db():
    """Load players in the DB but not in players_df (e.g., Saudi Pro League).
    Returned rows have null predictions and prediction_available=False.
    """
    trained_pks = set(int(pk) for pk in STATE['players_df']['player_pk'].tolist())
    rows = []
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT s.player_pk, s.short_name, s.long_name, s.league_name,
                       s.club_name, s.overall, s.age, s.nationality_name,
                       s.player_positions,
                       i.api_football_id, i.nationality,
                       (SELECT ps.position FROM player_stats ps
                        WHERE ps.player_pk = s.player_pk AND ps.position IS NOT NULL
                        ORDER BY ps.season DESC NULLS LAST LIMIT 1) AS stats_position
                FROM sofifa_attributes s
                LEFT JOIN player_identity i ON s.player_pk = i.player_pk
                WHERE s.league_name = ANY(%s)
                """,
                (list(UNTRAINED_LEAGUE_ALIASES.keys()),),
            )
            for r in cur.fetchall():
                (pk, short_name, long_name, league_name, club_name, overall, age,
                 nat_name, player_positions, api_id, nationality, stats_position) = r
                if int(pk) in trained_pks:
                    continue
                position = stats_position or _map_sofifa_position(player_positions) or 'Midfielder'
                rows.append({
                    'player_pk': int(pk),
                    'short_name': short_name or '',
                    'long_name': long_name or short_name or '',
                    'position': position,
                    'league_name': UNTRAINED_LEAGUE_ALIASES.get(league_name, league_name),
                    'club_name': club_name,
                    'nationality': nationality or nat_name,
                    'api_football_id': int(api_id) if api_id is not None else None,
                    'age': int(age) if age is not None else 0,
                    'overall': int(overall) if overall is not None else 0,
                    'gross_annual_eur': np.nan,
                    'predicted_low_eur': np.nan,
                    'predicted_high_eur': np.nan,
                    'predicted_center_eur': np.nan,
                    'prediction_error_pct': np.nan,
                    'actual_in_range': False,
                    'range_accuracy_result': False,
                    'prediction_available': False,
                })
            cur.close()
        finally:
            conn.close()
    except Exception as e:
        print(f"   WARNING: Could not load untrained players: {e}")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    print(f"   Loaded {len(df)} untrained players (e.g., Saudi Pro League)")
    return df


def combined_players_view():
    """Return players_df + untrained_df concatenated, for list/search endpoints."""
    untrained = STATE.get('untrained_df')
    if untrained is None or len(untrained) == 0:
        return STATE['players_df']
    return pd.concat([STATE['players_df'], untrained], ignore_index=True, sort=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global STATE
    print("Loading model artifacts...")
    STATE = load_artifacts()
    print(f"Loaded {len(STATE['players_df'])} players, {len(STATE['feature_names'])} features")
    enrich_players_df_from_db()
    STATE['players_df']['prediction_available'] = True
    STATE['untrained_df'] = load_untrained_players_from_db()
    yield
    STATE.clear()


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Football Salary Prediction API",
    description="Predict football player salary ranges using ML models with LLM-powered explanations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic Schemas
# ============================================================

class PredictionRequest(BaseModel):
    overall: float = Field(..., description="Player overall rating (1-99)")
    potential: float = Field(..., description="Potential overall rating")
    value_eur: float = Field(..., description="FIFA value estimate in EUR")
    wage_eur: float = Field(..., description="Weekly wage in EUR")
    age: float = Field(..., description="Player age")
    international_reputation: float = Field(..., description="International reputation (1-5)")
    shooting: float = Field(0, description="Shooting skill")
    passing: float = Field(0, description="Passing skill")
    dribbling: float = Field(0, description="Dribbling skill")
    defending: float = Field(0, description="Defending skill")
    physic: float = Field(0, description="Physical attribute")
    league_level: float = Field(1, description="League competitive level")
    movement_reactions: float = Field(0, description="Movement reactions")
    mentality_composure: float = Field(0, description="Mentality composure")
    release_clause_eur: float = Field(0, description="Release clause in EUR")
    appearances: float = Field(0, description="Number of appearances")
    minutes: float = Field(0, description="Total minutes played")
    rating: float = Field(0, description="Average match rating")
    goals: float = Field(0, description="Goals scored")
    assists: float = Field(0, description="Assists provided")
    market_value_eur: float = Field(0, description="Market value in EUR")
    preferred_foot: str = Field("Right", description="Preferred foot: 'Left' or 'Right'")
    position: str = Field("Attacker", description="Position: 'Attacker', 'Defender', 'Goalkeeper', 'Midfielder'")
    player_name: Optional[str] = Field(None, description="Player name (optional, for LLM context)")


class BatchPredictionRequest(BaseModel):
    player_pks: list[int] = Field(..., description="List of api_football_id values")


class SalaryRange(BaseModel):
    low_eur: float
    high_eur: float
    center_eur: float
    low_display: str
    high_display: str
    center_display: str
    range_width_pct: float


class PredictionResponse(BaseModel):
    predicted_range: SalaryRange
    actual_salary_eur: Optional[float] = None
    actual_salary_display: Optional[str] = None
    actual_in_range: Optional[bool] = None
    range_accuracy_result: Optional[bool] = None
    distance_to_nearest_bound_pct: Optional[float] = None
    percentile_rank: float
    feature_contributions: dict
    similar_players: list
    llm_explanation: str
    model_used: str


class PlayerSummary(BaseModel):
    player_pk: int
    short_name: str
    long_name: str
    position: str
    league_name: str
    age: int
    overall: int
    actual_salary_eur: float
    actual_salary_display: str
    predicted_low_eur: float
    predicted_high_eur: float
    predicted_center_eur: float
    predicted_range_display: str
    actual_in_range: bool
    range_accuracy_result: bool
    prediction_error_pct: float


# ============================================================
# Helper Functions
# ============================================================

def fmt_eur(value: float) -> str:
    """Format a number as EUR display string."""
    if value >= 1_000_000:
        return f"EUR {value:,.0f}"
    return f"EUR {value:,.0f}"


def engineer_features(data: dict) -> np.ndarray:
    """Replicate the feature engineering pipeline for a single player input."""
    s = STATE
    feature_names = s['feature_names']
    medians = s['feature_medians']

    # Start with base features
    row = {}

    # Numeric base features
    base_numeric = ['overall', 'potential', 'value_eur', 'wage_eur', 'age',
                    'international_reputation', 'shooting', 'passing', 'dribbling',
                    'defending', 'physic', 'league_level', 'movement_reactions',
                    'mentality_composure', 'release_clause_eur', 'appearances',
                    'minutes', 'rating', 'goals', 'assists', 'market_value_eur']

    for feat in base_numeric:
        val = data.get(feat, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            row[feat] = medians.get(feat, 0)
        else:
            row[feat] = float(val)

    # Encode preferred_foot
    foot = data.get('preferred_foot', 'Right')
    try:
        row['preferred_foot'] = int(s['le_foot'].transform([foot])[0])
    except (ValueError, KeyError):
        row['preferred_foot'] = int(s['le_foot'].transform(['Right'])[0])

    # Position one-hot (drop_first=True, Attacker is baseline)
    position = data.get('position', 'Attacker')
    for cat in s['position_categories']:
        col = f'pos_{cat}'
        if col in feature_names:
            row[col] = 1 if position == cat else 0

    # Log transforms
    row['log_value_eur'] = np.log1p(row['value_eur'])
    row['log_wage_eur'] = np.log1p(row['wage_eur'])
    row['log_market_value'] = np.log1p(row['market_value_eur'])
    row['log_release_clause'] = np.log1p(row['release_clause_eur'])

    # Interaction features
    minutes = row['minutes']
    row['goals_per_90'] = row['goals'] / (minutes / 90) if minutes > 0 else 0
    row['assists_per_90'] = row['assists'] / (minutes / 90) if minutes > 0 else 0
    row['goal_contributions'] = row['goals'] + row['assists']
    row['age_squared'] = row['age'] ** 2
    row['overall_x_reputation'] = row['overall'] * row['international_reputation']
    row['value_per_overall'] = row['value_eur'] / row['overall'] if row['overall'] > 0 else 0
    row['wage_to_value_ratio'] = row['wage_eur'] / row['value_eur'] if row['value_eur'] > 0 else 0

    # Replace inf
    for k, v in row.items():
        if np.isinf(v):
            row[k] = 0

    # Arrange in feature_names order
    arr = np.array([row.get(f, 0) for f in feature_names], dtype=np.float64).reshape(1, -1)
    return arr


def predict_range_single(features: np.ndarray) -> tuple:
    """Get range prediction from RF individual trees."""
    rf = STATE['rf_model']
    tree_preds = np.array([tree.predict(features) for tree in rf.estimators_])
    low = float(np.percentile(tree_preds, 25))
    high = float(np.percentile(tree_preds, 75))
    center = float(rf.predict(features)[0])
    return center, low, high


def compute_range_accuracy_result(actual: float, low: float, high: float, tolerance: float = 0.30) -> tuple:
    """Check if actual is within tolerance of range bounds. Returns (is_correct, distance_pct)."""
    if low <= actual <= high:
        return True, 0.0
    elif actual < low:
        dist = (low - actual) / actual if actual > 0 else 1.0
        return dist <= tolerance, round(dist * 100, 1)
    else:
        dist = (actual - high) / actual if actual > 0 else 1.0
        return dist <= tolerance, round(dist * 100, 1)


def find_similar_players(features: np.ndarray, exclude_pk: int = None, k: int = 5) -> list:
    """Find k most similar players using Euclidean distance in scaled feature space."""
    scaled = STATE['scaler'].transform(features)
    distances = np.linalg.norm(STATE['scaled_matrix'] - scaled, axis=1)
    df = STATE['players_df']

    indices = np.argsort(distances)
    results = []
    for idx in indices:
        if len(results) >= k:
            break
        row = df.iloc[idx]
        if exclude_pk is not None and int(row['player_pk']) == exclude_pk:
            continue
        results.append({
            'player_pk': int(row['player_pk']),
            'short_name': str(row['short_name']),
            'position': str(row['position']),
            'overall': int(row['overall']),
            'age': int(row['age']),
            'league_name': str(row['league_name']),
            'actual_salary_eur': float(row['gross_annual_eur']),
            'actual_salary_display': fmt_eur(row['gross_annual_eur']),
            'similarity_score': round(1.0 / (1.0 + float(distances[idx])), 3)
        })
    return results


def get_feature_contributions(features: np.ndarray) -> dict:
    """Get top feature contributions using RF feature importances weighted by feature values."""
    rf = STATE['rf_model']
    importances = rf.feature_importances_
    feature_names = STATE['feature_names']
    vals = features.flatten()

    # Scale feature values relative to mean
    mean_vals = np.mean(STATE['feature_matrix'], axis=0)
    std_vals = np.std(STATE['feature_matrix'], axis=0)
    std_vals[std_vals == 0] = 1

    # Contribution = importance * (value - mean) / std (directional)
    contributions = {}
    for i, name in enumerate(feature_names):
        contrib = float(importances[i])
        direction = "positive" if vals[i] >= mean_vals[i] else "negative"
        contributions[name] = {
            'importance': round(contrib, 4),
            'player_value': round(float(vals[i]), 2),
            'dataset_mean': round(float(mean_vals[i]), 2),
            'direction': direction
        }

    # Sort by importance and return top 10
    sorted_contribs = dict(sorted(contributions.items(), key=lambda x: x[1]['importance'], reverse=True)[:10])
    return sorted_contribs


def compute_percentile(salary_eur: float) -> float:
    """Compute percentile rank of a salary in the dataset."""
    salaries = STATE['players_df']['gross_annual_eur'].values
    return round(float(np.mean(salaries <= salary_eur) * 100), 1)


# ============================================================
# LLM Explanation (Google Gemini)
# ============================================================

# Human-readable translations for ML feature names
FEATURE_DISPLAY_NAMES = {
    'overall': 'overall quality rating',
    'potential': 'future potential rating',
    'value_eur': 'estimated market value',
    'wage_eur': 'current weekly wage',
    'age': 'age',
    'international_reputation': 'international reputation and brand recognition',
    'shooting': 'shooting ability',
    'passing': 'passing and vision',
    'dribbling': 'dribbling and ball control',
    'defending': 'defensive ability',
    'physic': 'physical strength and stamina',
    'league_level': 'league competitiveness',
    'movement_reactions': 'reaction speed and agility',
    'mentality_composure': 'composure under pressure',
    'release_clause_eur': 'contract release clause value',
    'appearances': 'number of match appearances',
    'minutes': 'total minutes on the pitch',
    'rating': 'average match rating',
    'goals': 'goals scored',
    'assists': 'assists provided',
    'market_value_eur': 'Transfermarkt market valuation',
    'preferred_foot': 'preferred foot',
    'log_value_eur': 'market value',
    'log_wage_eur': 'current wage level',
    'log_market_value': 'Transfermarkt valuation',
    'log_release_clause': 'release clause value',
    'goals_per_90': 'goal-scoring rate per 90 minutes',
    'assists_per_90': 'assist rate per 90 minutes',
    'goal_contributions': 'total goal contributions (goals + assists)',
    'age_squared': 'career stage and experience',
    'overall_x_reputation': 'combination of quality and global star power',
    'value_per_overall': 'market premium relative to their rating',
    'wage_to_value_ratio': 'wage relative to market value',
    'pos_Defender': 'playing as a defender',
    'pos_Goalkeeper': 'playing as a goalkeeper',
    'pos_Midfielder': 'playing as a midfielder',
}


def humanize_feature_name(name: str) -> str:
    """Convert a technical feature name to plain English."""
    return FEATURE_DISPLAY_NAMES.get(name, name.replace('_', ' '))


_gemini_model = None


def get_gemini_model():
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model

    api_key = os.getenv('GEMINI_API_KEY', '')
    if not api_key:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        _gemini_model = client
        return client
    except Exception:
        return None


_explanation_cache = {}


def generate_explanation(context: dict) -> str:
    """Generate LLM explanation for a salary prediction."""
    cache_key = str(context.get('player_pk', '')) or str(hash(json.dumps(context, default=str)))
    if cache_key in _explanation_cache:
        return _explanation_cache[cache_key]

    # Build optional sections for richer context
    positions_text = context.get('player_positions') or context.get('position') or 'N/A'
    traits_text = context.get('player_traits') or 'None listed'
    tags_text = context.get('player_tags') or 'None listed'
    club_text = context.get('club_name') or 'Unknown club'
    skill_moves = context.get('skill_moves')
    weak_foot = context.get('weak_foot')
    work_rate = context.get('work_rate')

    skill_line = ""
    if skill_moves or weak_foot:
        parts = []
        if skill_moves:
            parts.append(f"Skill Moves: {skill_moves}/5")
        if weak_foot:
            parts.append(f"Weak Foot: {weak_foot}/5")
        skill_line = ", ".join(parts)

    # Build the prompt for the LLM
    rep = context.get('international_reputation', 0)
    rep_desc = ""
    if rep and rep != 'N/A':
        rep = int(rep) if rep else 0
        if rep >= 5:
            rep_desc = "a global superstar with massive marketing and commercial appeal"
        elif rep >= 4:
            rep_desc = "a well-known name internationally — clubs value his brand and shirt sales potential"
        elif rep >= 3:
            rep_desc = "a recognized player at international level with growing commercial value"
        elif rep >= 2:
            rep_desc = "building his reputation beyond domestic football"
        else:
            rep_desc = "still establishing himself on the bigger stage"

    prompt = f"""You are a football journalist writing a short, engaging salary breakdown for fans. Imagine you're explaining to a friend over coffee why this player earns what he earns. Be specific, be fun, and sound like you actually watch football.

Here's the player profile:

Name: {context.get('player_name', 'Unknown')}
Age: {context.get('age', 'N/A')} | Club: {club_text} | League: {context.get('league', 'N/A')}
Can play: {positions_text}
Overall Rating: {context.get('overall', 'N/A')}/99
International Reputation: {rep_desc}
{f"Skill Moves: {skill_moves}/5 stars" if skill_moves else ""}
{f"Weak Foot: {weak_foot}/5 stars" if weak_foot else ""}
{f"Work Rate: {work_rate}" if work_rate else ""}

What makes him special:
- Traits: {traits_text}
- Known as: {tags_text}

Predicted Salary Range: {context.get('range_display', 'N/A')}
Actual Annual Salary: {context.get('actual_salary_display', 'Not available')}

What drives his value:
{context.get('top_factors', 'N/A')}

Players with a similar profile:
{context.get('similar_players_text', 'N/A')}

He earns more than {context.get('percentile', 'N/A')}% of players in the top 5 European leagues.

IMPORTANT RULES:
- Write 4-5 sentences MAX. Keep it tight.
- NEVER use technical ML terms like "feature", "model", "dataset", "prediction factor", "metric", or variable names.
- DO translate stats into real football meaning. For example:
  * High wage/value = "clubs have already invested heavily in him"
  * High overall rating = "he's one of the best in his position"
  * High reputation = "his name alone sells shirts and fills stadiums"
  * Can play multiple positions = "managers love him because he gives tactical flexibility — he can slot in at CB, CDM, or RB depending on the game plan"
  * Goal contributions = "he's directly involved in goals, which is what wins matches and trophies"
  * Age 27-30 = "he's in his prime years"
  * Age 33+ = "his experience and leadership still command respect"
- DO mention his traits and tags naturally — they're what make him unique on the pitch
- DO name the similar players and roughly what they earn to give context
- DO explain why he DESERVES this salary — what does he bring to the team that justifies this paycheck?
- Write like a real person who loves football, not like a computer report."""

    client = get_gemini_model()
    if client is None:
        explanation = _generate_template_explanation(context)
    else:
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            explanation = response.text.strip()
        except Exception as e:
            explanation = _generate_template_explanation(context)

    _explanation_cache[cache_key] = explanation
    return explanation


def _generate_template_explanation(ctx: dict) -> str:
    """Template-based fallback when LLM is unavailable."""
    name = ctx.get('player_name', 'This player')
    position = ctx.get('position', 'player')
    positions = ctx.get('player_positions') or position
    overall = ctx.get('overall', 'N/A')
    age = ctx.get('age', 'N/A')
    range_display = ctx.get('range_display', 'N/A')
    percentile = ctx.get('percentile', 'N/A')
    top_feature = ctx.get('top_feature_name', 'overall quality')
    club = ctx.get('club_name', '')
    traits = ctx.get('player_traits', '')
    tags = ctx.get('player_tags', '')
    rep = ctx.get('international_reputation', 0)

    # Age description
    try:
        age_val = float(age)
        if age_val <= 23:
            age_desc = f"At just {int(age_val)}, he's still developing"
        elif age_val <= 26:
            age_desc = f"At {int(age_val)}, he's entering his prime years"
        elif age_val <= 30:
            age_desc = f"At {int(age_val)}, he's right in his prime"
        elif age_val <= 33:
            age_desc = f"At {int(age_val)}, his experience and leadership are invaluable"
        else:
            age_desc = f"At {int(age_val)}, his veteran presence still commands respect"
    except (ValueError, TypeError):
        age_desc = f"At {age} years old"

    # Reputation description
    try:
        rep_val = int(rep) if rep else 0
    except (ValueError, TypeError):
        rep_val = 0

    parts = []

    # Opening sentence
    club_text = f" at {club}" if club else ""
    parts.append(
        f"{name}{club_text} is estimated to earn between {range_display} per year, "
        f"which puts him above {percentile}% of players in the top five European leagues."
    )

    # Age + overall + what makes him worth it
    overall_desc = ""
    try:
        ov = float(overall)
        if ov >= 85:
            overall_desc = " and rated as one of the elite players in European football"
        elif ov >= 80:
            overall_desc = " and rated as a high-quality player"
        elif ov >= 75:
            overall_desc = " and a solid contributor"
    except (ValueError, TypeError):
        pass

    parts.append(f"{age_desc}{overall_desc}, and his {top_feature} is the biggest factor behind that paycheck.")

    # Positional versatility
    if positions and ',' in str(positions):
        pos_list = [p.strip() for p in str(positions).split(',')]
        parts.append(
            f"Managers love his versatility — he can play {', '.join(pos_list[:-1])} or {pos_list[-1]}, "
            f"which makes him a tactical asset in any system."
        )

    # Traits
    if traits:
        trait_list = [t.strip() for t in traits.split(',')][:3]
        parts.append(
            f"On the pitch, he's known for being {', '.join(trait_list[:-1])} and {trait_list[-1]}, "
            f"which is exactly the kind of profile that clubs pay a premium for."
            if len(trait_list) > 1 else
            f"On the pitch, he's known for being {trait_list[0]}, "
            f"which is exactly the kind of profile that clubs pay a premium for."
        )

    # Tags + reputation
    if tags:
        tag_list = [t.strip().lstrip('#') for t in tags.split(',')][:2]
        tag_text = ' and '.join(tag_list)
        if rep_val >= 4:
            parts.append(
                f"Recognized as a {tag_text} with global star power, "
                f"his name alone brings commercial value and shirt sales to any club."
            )
        else:
            parts.append(f"Recognized as a {tag_text}, he brings real value to his team week in and week out.")
    elif rep_val >= 4:
        parts.append("Beyond the pitch, his global reputation brings commercial value and marketing appeal to his club.")

    return " ".join(parts)


def build_prediction_response(
    features: np.ndarray,
    player_name: str = None,
    actual_salary: float = None,
    player_pk: int = None,
    position: str = None,
    league: str = None,
    age: float = None,
    overall: float = None,
    international_reputation: float = None,
    player_positions: str = None,
    player_traits: str = None,
    player_tags: str = None,
    club_name: str = None,
    skill_moves: int = None,
    weak_foot: int = None,
    work_rate: str = None,
) -> dict:
    """Build a full prediction response with range, contributions, similar players, and LLM explanation."""

    # Range prediction
    center_log, low_log, high_log = predict_range_single(features)
    center_eur = float(np.expm1(center_log))
    low_eur = float(np.expm1(low_log))
    high_eur = float(np.expm1(high_log))

    range_width_pct = round((high_eur - low_eur) / center_eur * 100, 1) if center_eur > 0 else 0

    predicted_range = {
        'low_eur': round(low_eur, 0),
        'high_eur': round(high_eur, 0),
        'center_eur': round(center_eur, 0),
        'low_display': fmt_eur(low_eur),
        'high_display': fmt_eur(high_eur),
        'center_display': fmt_eur(center_eur),
        'range_width_pct': range_width_pct,
    }

    # Feature contributions
    contributions = get_feature_contributions(features)

    # Similar players
    similar = find_similar_players(features, exclude_pk=player_pk, k=5)

    # Percentile
    percentile = compute_percentile(center_eur)

    # Actual salary comparison
    actual_in_range = None
    range_accuracy_result = None
    distance_pct = None
    actual_display = None
    if actual_salary is not None and actual_salary > 0:
        actual_display = fmt_eur(actual_salary)
        actual_in_range = low_eur <= actual_salary <= high_eur
        range_accuracy_result, distance_pct = compute_range_accuracy_result(actual_salary, low_eur, high_eur)

    # LLM explanation context -- use human-readable feature names
    top_factors_list = list(contributions.items())[:5]
    top_factors_text = "\n".join([
        f"  - {humanize_feature_name(name)}: {'above' if v['direction'] == 'positive' else 'below'} average "
        f"(player: {v['player_value']}, league avg: {v['dataset_mean']})"
        for name, v in top_factors_list
    ])
    similar_text = "\n".join([
        f"  - {p['short_name']} ({p['position']}, Overall {p['overall']}): earns {p['actual_salary_display']}"
        for p in similar[:3]
    ])

    top_feature_raw = list(contributions.keys())[0] if contributions else 'overall'
    top_feature_human = humanize_feature_name(top_feature_raw)

    explanation = generate_explanation({
        'player_pk': player_pk,
        'player_name': player_name or 'Unknown Player',
        'age': age,
        'position': position,
        'league': league,
        'overall': overall,
        'international_reputation': international_reputation,
        'range_display': f"{fmt_eur(low_eur)} - {fmt_eur(high_eur)}",
        'actual_salary_display': actual_display or 'Not available',
        'top_factors': top_factors_text,
        'similar_players_text': similar_text,
        'percentile': percentile,
        'top_feature_name': top_feature_human,
        'player_positions': player_positions,
        'player_traits': player_traits,
        'player_tags': player_tags,
        'club_name': club_name,
        'skill_moves': skill_moves,
        'weak_foot': weak_foot,
        'work_rate': work_rate,
    })

    return {
        'predicted_range': predicted_range,
        'actual_salary_eur': actual_salary,
        'actual_salary_display': actual_display,
        'actual_in_range': actual_in_range,
        'range_accuracy_result': range_accuracy_result,
        'distance_to_nearest_bound_pct': distance_pct,
        'percentile_rank': percentile,
        'feature_contributions': contributions,
        'similar_players': similar,
        'llm_explanation': explanation,
        'model_used': STATE['model_config']['best_model_name'],
    }


# ============================================================
# Helper to get player row
# ============================================================

def get_player_row(player_pk: int) -> pd.Series:
    df = STATE['players_df']
    rows = df[df['player_pk'] == player_pk]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Player with pk={player_pk} not found")
    return rows.iloc[0]


def get_player_features(player_pk: int) -> np.ndarray:
    """Get the pre-computed feature vector for a player."""
    df = STATE['players_df']
    idx = df.index[df['player_pk'] == player_pk]
    if len(idx) == 0:
        raise HTTPException(status_code=404, detail=f"Player with pk={player_pk} not found")
    return STATE['feature_matrix'][idx[0]].reshape(1, -1)


def _clean_optional(val):
    if val is None or pd.isna(val):
        return None
    return val


def player_to_summary(row) -> dict:
    api_football_id = _clean_optional(row.get('api_football_id'))
    nationality = _clean_optional(row.get('nationality'))
    club_name = _clean_optional(row.get('club_name'))
    actual_raw = _clean_optional(row.get('gross_annual_eur'))
    low_raw = _clean_optional(row.get('predicted_low_eur'))
    high_raw = _clean_optional(row.get('predicted_high_eur'))
    center_raw = _clean_optional(row.get('predicted_center_eur'))
    error_raw = _clean_optional(row.get('prediction_error_pct'))
    prediction_available = bool(row.get('prediction_available', True)) and low_raw is not None and high_raw is not None

    if not prediction_available:
        status = 'prediction_not_available'
    elif actual_raw is None:
        status = 'unknown'
    elif float(actual_raw) > float(high_raw):
        status = 'overpaid'
    elif float(actual_raw) < float(low_raw):
        status = 'underpaid'
    else:
        status = 'fair'

    return {
        'player_pk': int(row['player_pk']),
        'api_football_id': int(api_football_id) if api_football_id is not None else None,
        'short_name': str(row['short_name']),
        'long_name': str(row['long_name']),
        'position': str(row['position']),
        'league_name': str(row['league_name']),
        'club_name': str(club_name) if club_name is not None else None,
        'nationality': str(nationality) if nationality is not None else None,
        'age': int(row['age']) if _clean_optional(row.get('age')) is not None else None,
        'overall': int(row['overall']) if _clean_optional(row.get('overall')) is not None else None,
        'actual_salary_eur': float(actual_raw) if actual_raw is not None else None,
        'actual_salary_display': fmt_eur(actual_raw) if actual_raw is not None else None,
        'predicted_low_eur': round(float(low_raw), 0) if low_raw is not None else None,
        'predicted_high_eur': round(float(high_raw), 0) if high_raw is not None else None,
        'predicted_center_eur': round(float(center_raw), 0) if center_raw is not None else None,
        'predicted_range_display': (
            f"{fmt_eur(low_raw)} - {fmt_eur(high_raw)}"
            if low_raw is not None and high_raw is not None else None
        ),
        'actual_in_range': bool(row['actual_in_range']) if _clean_optional(row.get('actual_in_range')) is not None else False,
        'range_accuracy_result': bool(row['range_accuracy_result']) if _clean_optional(row.get('range_accuracy_result')) is not None else False,
        'prediction_error_pct': round(float(error_raw), 1) if error_raw is not None else None,
        'prediction_available': prediction_available,
        'status': status,
    }


# ============================================================
# API-FOOTBALL INTEGRATION
# ============================================================

APIFOOTBALL_BASE = "https://v3.football.api-sports.io"


def fetch_apifootball(endpoint: str, params: dict) -> dict:
    """Fetch data from API-Football v3."""
    if not FOOTBALL_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="FOOTBALL_API_KEY is not configured. Add it to your .env file."
        )

    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{APIFOOTBALL_BASE}/{endpoint}?{query}"

    req = urllib.request.Request(url)
    req.add_header("x-apisports-key", FOOTBALL_API_KEY)
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=e.code, detail=f"API-Football error: {e.reason}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach API-Football: {str(e)}")


def fetch_player_from_api(api_football_id: int, season: int = 2026) -> dict:
    """Fetch player info + stats from API-Football and return structured data."""
    data = fetch_apifootball("players", {"id": api_football_id, "season": season})

    results = data.get("results", 0)
    if results == 0 or not data.get("response"):
        # Try previous season
        if season > 2020:
            data = fetch_apifootball("players", {"id": api_football_id, "season": season - 1})
            results = data.get("results", 0)

    if results == 0 or not data.get("response"):
        raise HTTPException(
            status_code=404,
            detail=f"Player with api_football_id={api_football_id} not found on API-Football for season {season}. "
                   f"Verify the ID is correct at api-football.com."
        )

    entry = data["response"][0]
    player_info = entry.get("player", {})

    # Aggregate stats across all leagues in the season
    total_appearances = 0
    total_minutes = 0
    total_goals = 0
    total_assists = 0
    ratings = []
    position = None
    league_name = None

    for stat_block in entry.get("statistics", []):
        games = stat_block.get("games", {})
        goals_data = stat_block.get("goals", {})
        league_data = stat_block.get("league", {})

        apps = games.get("appearances") or 0
        mins = games.get("minutes") or 0
        g = goals_data.get("total") or 0
        a = goals_data.get("assists") or 0
        r = games.get("rating")

        total_appearances += apps
        total_minutes += mins
        total_goals += g
        total_assists += a
        if r:
            try:
                ratings.append(float(r))
            except (ValueError, TypeError):
                pass

        # Use position and league from the block with most appearances
        if apps > 0:
            if position is None or apps > (stat_block.get("_best_apps", 0)):
                position = games.get("position")
                league_name = league_data.get("name")
                stat_block["_best_apps"] = apps

    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0.0

    # If minutes > 0 but appearances is 0, estimate appearances from minutes
    if total_appearances == 0 and total_minutes > 0:
        total_appearances = max(1, round(total_minutes / 80))

    # Normalize position to match our model's categories
    pos_map = {
        "Attacker": "Attacker", "Midfielder": "Midfielder",
        "Defender": "Defender", "Goalkeeper": "Goalkeeper",
    }
    position = pos_map.get(position, "Midfielder")

    return {
        "api_football_id": api_football_id,
        "name": player_info.get("name", "Unknown"),
        "firstname": player_info.get("firstname", ""),
        "lastname": player_info.get("lastname", ""),
        "age": player_info.get("age", 0),
        "nationality": player_info.get("nationality", ""),
        "height": player_info.get("height", ""),
        "weight": player_info.get("weight", ""),
        "photo": player_info.get("photo", ""),
        "season": season,
        "position": position,
        "league_name": league_name,
        "appearances": total_appearances,
        "minutes": total_minutes,
        "goals": total_goals,
        "assists": total_assists,
        "rating": avg_rating,
    }


def fetch_past_season_appearances(api_football_id: int) -> dict:
    """Fetch total appearances from the player_stats table (past seasons stored in DB)."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT season, SUM(appearances) as total_apps, SUM(minutes) as total_mins
               FROM player_stats
               WHERE api_football_id = %s
               GROUP BY season
               ORDER BY season DESC""",
            (api_football_id,)
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    if not rows:
        return {"total_appearances": 0, "seasons": [], "per_season": []}

    total = sum(r[1] or 0 for r in rows)
    per_season = [{"season": r[0], "appearances": r[1] or 0, "minutes": r[2] or 0} for r in rows]
    return {
        "total_appearances": total,
        "seasons": [r[0] for r in rows],
        "per_season": per_season,
    }


def lookup_sofifa_by_api_id(api_football_id: int) -> dict:
    """Look up FIFA attributes and market value from the database using api_football_id."""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Find player_pk from player_identity
        cur.execute(
            "SELECT player_pk, canonical_name FROM player_identity WHERE api_football_id = %s",
            (api_football_id,)
        )
        row = cur.fetchone()

        if row is None:
            cur.close()
            return None

        player_pk = row["player_pk"]

        # Get sofifa attributes
        cur.execute(
            """SELECT overall, potential, value_eur, wage_eur, age,
                      international_reputation, shooting, passing, dribbling,
                      defending, physic, league_level, movement_reactions,
                      mentality_composure, release_clause_eur, preferred_foot,
                      player_positions, player_traits, player_tags, club_name,
                      skill_moves, weak_foot, work_rate
               FROM sofifa_attributes WHERE player_pk = %s""",
            (player_pk,)
        )
        sofifa = cur.fetchone()

        # Get market value
        cur.execute(
            "SELECT market_value_eur FROM market_values WHERE player_pk = %s",
            (player_pk,)
        )
        market = cur.fetchone()

        # Get salary if available
        cur.execute(
            "SELECT gross_annual_eur FROM salaries WHERE player_pk = %s",
            (player_pk,)
        )
        salary = cur.fetchone()

        cur.close()
    finally:
        conn.close()

    if sofifa is None:
        return None

    result = dict(sofifa)
    result["player_pk"] = player_pk
    result["market_value_eur"] = market["market_value_eur"] if market else 0
    result["actual_salary_eur"] = salary["gross_annual_eur"] if salary else None
    return result


# ============================================================
# PREDICTION ENDPOINTS
# ============================================================

@app.post("/api/predict", response_model=None, tags=["Prediction"])
def predict_salary(req: PredictionRequest):
    """Predict salary range for a player given raw attributes."""
    features = engineer_features(req.model_dump())
    result = build_prediction_response(
        features,
        player_name=req.player_name,
        position=req.position,
        age=req.age,
        overall=req.overall,
        international_reputation=req.international_reputation,
    )
    return result


@app.get("/api/predict/live/{api_football_id}", tags=["Prediction"])
def predict_live(
    api_football_id: int,
    season: int = Query(2026, description="Football season year (e.g., 2026 for 2026-2027)")
):
    """Predict salary range using LIVE data from API-Football.

    Fetches real-time player stats from API-Football, combines with
    FIFA attributes from the database, and returns a salary range prediction
    with LLM explanation.

    Requires FOOTBALL_API_KEY in .env file.
    """
    warnings = []

    # 1. Fetch live stats from API-Football
    api_data = fetch_player_from_api(api_football_id, season)

    # 2. Check appearances threshold (current season + past seasons)
    past_data = fetch_past_season_appearances(api_football_id)

    if api_data["appearances"] < 20:
        warnings.append(
            f"Current season warning: player has only {api_data['appearances']} appearances "
            f"in the {season} season (minimum 20 recommended for reliable prediction). "
            f"The prediction may be less accurate."
        )

    total_historical_apps = past_data["total_appearances"]
    if total_historical_apps < 20:
        seasons_text = ", ".join(str(s) for s in past_data["seasons"]) if past_data["seasons"] else "none"
        warnings.append(
            f"Historical data warning: player has only {total_historical_apps} total appearances "
            f"across past seasons ({seasons_text}). Limited historical data may affect prediction reliability."
        )

    if api_data["appearances"] == 0 and total_historical_apps == 0:
        return {
            "error": True,
            "message": f"Player {api_data['name']} has 0 appearances in both the {season} season "
                       f"and past seasons. Cannot make a prediction without match data.",
            "player_info": api_data,
        }

    if api_data["appearances"] == 0:
        return {
            "error": True,
            "message": f"Player {api_data['name']} has 0 appearances in the {season} season. "
                       f"Cannot make a prediction without current season match data.",
            "player_info": api_data,
            "historical_appearances": total_historical_apps,
        }

    # 3. Look up FIFA attributes from database
    sofifa_data = lookup_sofifa_by_api_id(api_football_id)

    if sofifa_data is None:
        return {
            "error": True,
            "message": (
                f"Player {api_data['name']} (api_football_id={api_football_id}) was found on "
                f"API-Football but does not exist in our FIFA attributes database. "
                f"The prediction model requires FIFA attributes (overall rating, potential, "
                f"shooting, passing, dribbling, defending, physic, etc.) which are not "
                f"available from API-Football alone. This player cannot be predicted."
            ),
            "player_info": api_data,
            "available_stats": {
                "appearances": api_data["appearances"],
                "goals": api_data["goals"],
                "assists": api_data["assists"],
                "rating": api_data["rating"],
                "minutes": api_data["minutes"],
            },
            "missing_data": [
                "overall", "potential", "value_eur", "wage_eur",
                "international_reputation", "shooting", "passing",
                "dribbling", "defending", "physic", "league_level",
                "movement_reactions", "mentality_composure",
                "release_clause_eur", "market_value_eur"
            ],
        }

    # 4. Combine live stats with FIFA attributes
    combined = {
        # FIFA attributes from database
        "overall": sofifa_data.get("overall", 0),
        "potential": sofifa_data.get("potential", 0),
        "value_eur": sofifa_data.get("value_eur", 0),
        "wage_eur": sofifa_data.get("wage_eur", 0),
        "age": api_data["age"] or sofifa_data.get("age", 0),
        "international_reputation": sofifa_data.get("international_reputation", 0),
        "shooting": sofifa_data.get("shooting", 0),
        "passing": sofifa_data.get("passing", 0),
        "dribbling": sofifa_data.get("dribbling", 0),
        "defending": sofifa_data.get("defending", 0),
        "physic": sofifa_data.get("physic", 0),
        "league_level": sofifa_data.get("league_level", 1),
        "movement_reactions": sofifa_data.get("movement_reactions", 0),
        "mentality_composure": sofifa_data.get("mentality_composure", 0),
        "release_clause_eur": sofifa_data.get("release_clause_eur", 0),
        "market_value_eur": sofifa_data.get("market_value_eur", 0),
        # Live stats from API-Football
        "appearances": api_data["appearances"],
        "minutes": api_data["minutes"],
        "rating": api_data["rating"],
        "goals": api_data["goals"],
        "assists": api_data["assists"],
        # Categoricals
        "preferred_foot": sofifa_data.get("preferred_foot", "Right"),
        "position": api_data["position"],
    }

    # 5. Engineer features and predict
    features = engineer_features(combined)
    result = build_prediction_response(
        features,
        player_name=api_data["name"],
        actual_salary=sofifa_data.get("actual_salary_eur"),
        player_pk=sofifa_data.get("player_pk"),
        position=api_data["position"],
        league=api_data["league_name"],
        age=api_data["age"],
        overall=sofifa_data.get("overall"),
        international_reputation=sofifa_data.get("international_reputation"),
        player_positions=sofifa_data.get("player_positions"),
        player_traits=sofifa_data.get("player_traits"),
        player_tags=sofifa_data.get("player_tags"),
        club_name=sofifa_data.get("club_name"),
        skill_moves=sofifa_data.get("skill_moves"),
        weak_foot=sofifa_data.get("weak_foot"),
        work_rate=sofifa_data.get("work_rate"),
    )

    # 6. Add live data and warnings to response
    result["data_source"] = {
        "stats": "API-Football (live)",
        "fifa_attributes": "database",
        "season": season,
    }
    result["player_info"] = api_data
    if warnings:
        result["warnings"] = warnings

    return result


@app.get("/api/predict/{player_pk}", tags=["Prediction"])
def predict_player(player_pk: int):
    """Predict salary range for a player in the database."""
    untrained = STATE.get('untrained_df')
    if untrained is not None and len(untrained) and (untrained['player_pk'] == player_pk).any():
        urow = untrained[untrained['player_pk'] == player_pk].iloc[0]
        api_id = urow.get('api_football_id')
        has_id = pd.notna(api_id)
        raise HTTPException(
            status_code=409,
            detail={
                'message': (
                    f"Player {urow['short_name']} is in {urow['league_name']} which is outside "
                    f"the trained dataset. Use /api/predict/live/{int(api_id)} for a live prediction."
                ) if has_id else (
                    f"Player {urow['short_name']} is not in the trained dataset and has no api_football_id."
                ),
                'api_football_id': int(api_id) if has_id else None,
                'live_endpoint': f"/api/predict/live/{int(api_id)}" if has_id else None,
            },
        )
    row = get_player_row(player_pk)
    features = get_player_features(player_pk)

    # Fetch extra personality fields from sofifa_attributes
    extra = {}
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT player_positions, player_traits, player_tags, club_name,
                          skill_moves, weak_foot, work_rate
                   FROM sofifa_attributes WHERE player_pk = %s""",
                (player_pk,)
            )
            sofifa_row = cur.fetchone()
            cur.close()
        finally:
            conn.close()
        if sofifa_row:
            extra = {
                'player_positions': sofifa_row[0],
                'player_traits': sofifa_row[1],
                'player_tags': sofifa_row[2],
                'club_name': sofifa_row[3],
                'skill_moves': sofifa_row[4],
                'weak_foot': sofifa_row[5],
                'work_rate': sofifa_row[6],
            }
    except Exception:
        pass

    result = build_prediction_response(
        features,
        player_name=str(row['short_name']),
        actual_salary=float(row['gross_annual_eur']),
        player_pk=player_pk,
        position=str(row['position']),
        league=str(row['league_name']),
        age=float(row['age']),
        overall=float(row['overall']),
        international_reputation=float(row.get('international_reputation', 0)),
        **extra,
    )
    return result


@app.post("/api/predict/batch", tags=["Prediction"])
def predict_batch(req: BatchPredictionRequest):
    """Batch predict salary ranges for multiple players by api_football_id (live)."""
    if not FOOTBALL_API_KEY:
        raise HTTPException(status_code=503, detail="FOOTBALL_API_KEY not configured in .env")

    results = []
    for api_id in req.player_pks:
        try:
            resp = predict_live(api_id, season=2026)
            results.append(resp)
        except HTTPException as e:
            results.append({'error': True, 'api_football_id': api_id, 'message': e.detail})
    return results


# ============================================================
# PLAYER ENDPOINTS
# ============================================================

@app.get("/api/players", tags=["Players"])
def list_players(
    league: Optional[str] = Query(None, description="Filter by league name (substring match)"),
    position: Optional[str] = Query(None, description="Filter by position (substring match)"),
    q: Optional[str] = Query(None, description="Substring match against short_name, long_name, club_name"),
    club: Optional[str] = Query(None, description="Exact match on club_name"),
    nationality: Optional[str] = Query(None, description="Exact match on nationality"),
    max_salary_eur: Optional[float] = Query(None, description="Include only players with actual_salary_eur <= this value"),
    status: Optional[str] = Query(None, description="Filter by status: overpaid | fair | underpaid"),
    sort_by: str = Query("actual_salary", description="Sort field: actual_salary, overall, age, prediction_error"),
    order: str = Query("desc", description="Sort order: asc or desc"),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=1000),
):
    """List all players with predicted ranges and actual salaries."""
    df = combined_players_view().copy()

    if league:
        df = df[df['league_name'].str.contains(league, case=False, na=False)]
    if position:
        df = df[df['position'].str.contains(position, case=False, na=False)]
    if q:
        q_lower = q.lower()
        mask = (
            df['short_name'].astype(str).str.lower().str.contains(q_lower, na=False)
            | df['long_name'].astype(str).str.lower().str.contains(q_lower, na=False)
        )
        if 'club_name' in df.columns:
            mask = mask | df['club_name'].astype(str).str.lower().str.contains(q_lower, na=False)
        df = df[mask]
    if club and 'club_name' in df.columns:
        df = df[df['club_name'] == club]
    if nationality and 'nationality' in df.columns:
        df = df[df['nationality'] == nationality]
    if max_salary_eur is not None:
        df = df[df['gross_annual_eur'].fillna(0) <= max_salary_eur]
    if status:
        s = status.lower()
        if s not in ('overpaid', 'fair', 'underpaid'):
            raise HTTPException(
                status_code=400,
                detail="status must be one of: overpaid, fair, underpaid",
            )
        if 'prediction_available' in df.columns:
            df = df[df['prediction_available'] == True]
        if s == 'overpaid':
            df = df[df['gross_annual_eur'] > df['predicted_high_eur']]
        elif s == 'underpaid':
            df = df[df['gross_annual_eur'] < df['predicted_low_eur']]
        else:
            df = df[(df['gross_annual_eur'] >= df['predicted_low_eur'])
                    & (df['gross_annual_eur'] <= df['predicted_high_eur'])]

    sort_map = {
        'actual_salary': 'gross_annual_eur',
        'overall': 'overall',
        'age': 'age',
        'prediction_error': 'prediction_error_pct',
        'predicted_salary': 'predicted_center_eur',
    }
    sort_col = sort_map.get(sort_by, 'gross_annual_eur')
    ascending = order.lower() == 'asc'
    df = df.sort_values(sort_col, ascending=ascending)

    total = len(df)
    start = (page - 1) * per_page
    end = start + per_page
    page_df = df.iloc[start:end]

    players = [player_to_summary(row) for _, row in page_df.iterrows()]

    return {
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page,
        'players': players,
    }


@app.get("/api/players/search", tags=["Players"])
def search_players(q: str = Query(..., min_length=1, description="Search query")):
    """Search players by name (fuzzy match). Returns api_football_id for use with /api/predict/live/."""
    df = combined_players_view()
    q_lower = q.lower()
    mask = (
        df['short_name'].astype(str).str.lower().str.contains(q_lower, na=False) |
        df['long_name'].astype(str).str.lower().str.contains(q_lower, na=False)
    )
    results = df[mask].head(20)
    return [player_to_summary(row) for _, row in results.iterrows()]


@app.get("/api/players/lookup", tags=["Players"])
def lookup_api_football_id(q: str = Query(..., min_length=1, description="Player name to look up")):
    """Look up api_football_id by player name. Use this ID with /api/predict/live/{id}."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT pi.api_football_id, pi.canonical_name, pi.nationality, pi.player_pk
               FROM player_identity pi
               WHERE pi.canonical_name ILIKE %s AND pi.api_football_id IS NOT NULL
               LIMIT 20""",
            (f"%{q}%",)
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No players found matching '{q}'")

    return [{
        'api_football_id': r[0],
        'name': r[1],
        'nationality': r[2],
        'player_pk': r[3],
        'predict_url': f'/api/predict/live/{r[0]}',
    } for r in rows]


@app.get("/api/players/search-api", tags=["Players"])
def search_api_football(q: str = Query(..., min_length=3, description="Player name to search on API-Football")):
    """Search for a player on API-Football by name. Returns api_football_id.
    Requires FOOTBALL_API_KEY."""
    data = fetch_apifootball("players/profiles", {"search": q})
    results = data.get("response", [])
    if not results:
        raise HTTPException(status_code=404, detail=f"No players found on API-Football matching '{q}'")

    return [{
        'api_football_id': r.get("player", {}).get("id"),
        'name': r.get("player", {}).get("name"),
        'firstname': r.get("player", {}).get("firstname"),
        'lastname': r.get("player", {}).get("lastname"),
        'age': r.get("player", {}).get("age"),
        'nationality': r.get("player", {}).get("nationality"),
        'photo': r.get("player", {}).get("photo"),
        'predict_url': f'/api/predict/live/{r.get("player", {}).get("id")}',
    } for r in results[:10]]


@app.get("/api/players/top-overpaid", tags=["Players"])
def top_overpaid(limit: int = Query(20, ge=1, le=100)):
    """Players with actual salary most above predicted range (overpaid)."""
    df = STATE['players_df'].copy()
    df['overpaid_amount'] = df['gross_annual_eur'] - df['predicted_high_eur']
    df = df[df['overpaid_amount'] > 0].sort_values('overpaid_amount', ascending=False).head(limit)
    return [{
        **player_to_summary(row),
        'overpaid_amount_eur': round(float(row['overpaid_amount']), 0),
        'overpaid_display': fmt_eur(row['overpaid_amount']),
    } for _, row in df.iterrows()]


@app.get("/api/players/top-underpaid", tags=["Players"])
def top_underpaid(limit: int = Query(20, ge=1, le=100)):
    """Players with actual salary most below predicted range (underpaid)."""
    df = STATE['players_df'].copy()
    df['underpaid_amount'] = df['predicted_low_eur'] - df['gross_annual_eur']
    df = df[df['underpaid_amount'] > 0].sort_values('underpaid_amount', ascending=False).head(limit)
    return [{
        **player_to_summary(row),
        'underpaid_amount_eur': round(float(row['underpaid_amount']), 0),
        'underpaid_display': fmt_eur(row['underpaid_amount']),
    } for _, row in df.iterrows()]


@app.get("/api/players/{player_pk}", tags=["Players"])
def get_player(player_pk: int):
    """Get full details for a single player."""
    untrained = STATE.get('untrained_df')
    if untrained is not None and len(untrained) and (untrained['player_pk'] == player_pk).any():
        return player_to_summary(untrained[untrained['player_pk'] == player_pk].iloc[0])
    row = get_player_row(player_pk)
    summary = player_to_summary(row)

    # Add extra details
    extra_fields = ['potential', 'value_eur', 'wage_eur', 'market_value_eur',
                    'appearances', 'goals', 'assists', 'rating', 'shooting', 'passing',
                    'dribbling', 'defending', 'physic', 'international_reputation',
                    'movement_reactions', 'mentality_composure', 'release_clause_eur',
                    'minutes', 'league_level', 'preferred_foot']
    for f in extra_fields:
        if f in row.index:
            val = row[f]
            if pd.notna(val):
                summary[f] = float(val) if isinstance(val, (np.floating, float)) else val

    return summary


# ============================================================
# ANALYTICS ENDPOINTS
# ============================================================

@app.get("/api/analytics/overview", tags=["Analytics"])
def dataset_overview():
    """Get dataset overview statistics."""
    stats = STATE['dataset_stats']
    config = STATE['model_config']
    return {
        'total_players': stats['total_players'],
        'total_features': config['n_features'],
        'train_size': config['train_size'],
        'test_size': config['test_size'],
        'salary_stats': {
            'mean': round(stats['salary_mean'], 0),
            'median': round(stats['salary_median'], 0),
            'min': round(stats['salary_min'], 0),
            'max': round(stats['salary_max'], 0),
            'std': round(stats['salary_std'], 0),
            'mean_display': fmt_eur(stats['salary_mean']),
            'median_display': fmt_eur(stats['salary_median']),
        },
        'leagues': stats['leagues'],
        'positions': stats['positions'],
        'age_stats': stats['age_stats'],
        'model_performance': stats['final_metrics'],
    }


@app.get("/api/analytics/leagues", tags=["Analytics"])
def league_stats():
    """Salary statistics by league."""
    df = combined_players_view()
    result = []
    for league in sorted(df['league_name'].dropna().unique()):
        ldf = df[df['league_name'] == league]
        salaries = ldf['gross_annual_eur'].dropna()
        has_salary = len(salaries) > 0
        entry = {
            'league': league,
            'player_count': int(len(ldf)),
            'has_salary_data': has_salary,
            'mean_salary': round(float(salaries.mean()), 0) if has_salary else None,
            'median_salary': round(float(salaries.median()), 0) if has_salary else None,
            'min_salary': round(float(salaries.min()), 0) if has_salary else None,
            'max_salary': round(float(salaries.max()), 0) if has_salary else None,
            'std_salary': round(float(salaries.std()), 0) if has_salary and len(salaries) > 1 else None,
            'mean_overall': round(float(ldf['overall'].dropna().mean()), 1) if ldf['overall'].notna().any() else None,
            'top_paid_player': None,
            'top_paid_salary': None,
            'range_accuracy_pct': (
                round(float(ldf['range_accuracy_result'].mean() * 100), 1)
                if has_salary and 'range_accuracy_result' in ldf.columns else None
            ),
            'mean_salary_display': fmt_eur(salaries.mean()) if has_salary else None,
        }
        if has_salary:
            top_row = ldf.loc[ldf['gross_annual_eur'].idxmax()]
            entry['top_paid_player'] = str(top_row['short_name'])
            entry['top_paid_salary'] = round(float(top_row['gross_annual_eur']), 0)
        result.append(entry)
    return sorted(result, key=lambda x: (x['median_salary'] or -1), reverse=True)


@app.get("/api/analytics/positions", tags=["Analytics"])
def position_stats():
    """Salary statistics by position."""
    df = STATE['players_df']
    result = []
    for pos in sorted(df['position'].dropna().unique()):
        pdf = df[df['position'] == pos]
        result.append({
            'position': pos,
            'player_count': int(len(pdf)),
            'mean_salary': round(float(pdf['gross_annual_eur'].mean()), 0),
            'median_salary': round(float(pdf['gross_annual_eur'].median()), 0),
            'min_salary': round(float(pdf['gross_annual_eur'].min()), 0),
            'max_salary': round(float(pdf['gross_annual_eur'].max()), 0),
            'mean_overall': round(float(pdf['overall'].mean()), 1),
            'mean_age': round(float(pdf['age'].mean()), 1),
            'range_accuracy_pct': round(float(pdf['range_accuracy_result'].mean() * 100), 1),
            'mean_salary_display': fmt_eur(pdf['gross_annual_eur'].mean()),
        })
    return sorted(result, key=lambda x: x['median_salary'], reverse=True)


@app.get("/api/analytics/age-analysis", tags=["Analytics"])
def age_analysis():
    """Salary by age buckets and peak earning age."""
    df = STATE['players_df']

    # Age buckets
    bins = [(17, 21), (21, 24), (24, 27), (27, 30), (30, 33), (33, 40)]
    buckets = []
    for low, high in bins:
        bucket_df = df[(df['age'] >= low) & (df['age'] < high)]
        if len(bucket_df) > 0:
            buckets.append({
                'age_range': f"{low}-{high-1}",
                'player_count': int(len(bucket_df)),
                'mean_salary': round(float(bucket_df['gross_annual_eur'].mean()), 0),
                'median_salary': round(float(bucket_df['gross_annual_eur'].median()), 0),
                'mean_overall': round(float(bucket_df['overall'].mean()), 1),
                'mean_salary_display': fmt_eur(bucket_df['gross_annual_eur'].mean()),
            })

    # Peak earning age
    age_salary = df.groupby('age')['gross_annual_eur'].mean()
    peak_age = int(age_salary.idxmax())

    return {
        'age_buckets': buckets,
        'peak_earning_age': peak_age,
        'peak_earning_salary': round(float(age_salary.max()), 0),
        'peak_earning_salary_display': fmt_eur(age_salary.max()),
    }


@app.get("/api/analytics/salary-factors", tags=["Analytics"])
def salary_factors():
    """Top feature correlations with salary."""
    importances = STATE['feature_importances']
    rf_imp = importances.get('rf_model', importances.get('best_model', {}))
    sorted_imp = sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)
    return [{
        'rank': i + 1,
        'feature': name,
        'importance': round(val, 4),
    } for i, (name, val) in enumerate(sorted_imp)]


@app.get("/api/analytics/compare", tags=["Analytics"])
def compare_players(pks: str = Query(..., description="Comma-separated player_pk values")):
    """Compare players side by side."""
    pk_list = [int(x.strip()) for x in pks.split(',')]
    if len(pk_list) < 2 or len(pk_list) > 10:
        raise HTTPException(status_code=400, detail="Provide 2-10 player PKs")

    results = []
    for pk in pk_list:
        row = get_player_row(pk)
        summary = player_to_summary(row)
        attrs = ['shooting', 'passing', 'dribbling', 'defending', 'physic',
                 'potential', 'value_eur', 'wage_eur', 'market_value_eur',
                 'appearances', 'goals', 'assists', 'rating', 'international_reputation']
        for a in attrs:
            if a in row.index and pd.notna(row[a]):
                summary[a] = float(row[a]) if isinstance(row[a], (np.floating, float)) else row[a]
        results.append(summary)

    return {'players': results}


# ============================================================
# MODEL ENDPOINTS
# ============================================================

@app.get("/api/model/metrics", tags=["Model"])
def model_metrics():
    """Performance metrics for all trained models."""
    return STATE['model_metrics']


@app.get("/api/model/feature-importances", tags=["Model"])
def feature_importances():
    """Ranked feature importances."""
    importances = STATE['feature_importances']
    rf_imp = importances.get('rf_model', {})
    best_imp = importances.get('best_model', {})

    sorted_rf = sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)
    sorted_best = sorted(best_imp.items(), key=lambda x: x[1], reverse=True)

    return {
        'rf_model_importances': [{'rank': i+1, 'feature': n, 'importance': round(v, 4)}
                                  for i, (n, v) in enumerate(sorted_rf)],
        'best_model_importances': [{'rank': i+1, 'feature': n, 'importance': round(v, 4)}
                                    for i, (n, v) in enumerate(sorted_best)],
    }


@app.get("/api/model/summary", tags=["Model"])
def model_summary():
    """Summary of the best model."""
    config = STATE['model_config']
    stats = STATE['dataset_stats']
    importances = STATE['feature_importances']
    rf_imp = importances.get('rf_model', importances.get('best_model', {}))
    top5 = sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        'best_model_name': config['best_model_name'],
        'n_features': config['n_features'],
        'n_players': config['n_players'],
        'train_size': config['train_size'],
        'test_size': config['test_size'],
        'performance': stats['final_metrics'],
        'top_5_features': [{'feature': n, 'importance': round(v, 4)} for n, v in top5],
        'hyperparameters': {
            'rf': config.get('rf_best_params', {}),
            'xgb': config.get('xgb_best_params', {}),
            'gb': config.get('gb_best_params', {}),
        },
    }


# ============================================================
# VISUALIZATION ENDPOINTS
# ============================================================

@app.get("/api/visualizations", tags=["Visualizations"])
def list_visualizations():
    """List all available graphs."""
    graphs = []

    # Original figures
    if os.path.isdir(FIGURES_DIR):
        for f in sorted(os.listdir(FIGURES_DIR)):
            if f.endswith('.png'):
                name = f.replace('.png', '')
                graphs.append({
                    'name': name,
                    'filename': f,
                    'category': 'original',
                    'url': f'/api/visualizations/{name}',
                })

    # API-generated figures
    if os.path.isdir(FIGURES_API_DIR):
        for f in sorted(os.listdir(FIGURES_API_DIR)):
            if f.endswith('.png'):
                name = f.replace('.png', '')
                graphs.append({
                    'name': name,
                    'filename': f,
                    'category': 'academic',
                    'url': f'/api/visualizations/{name}',
                })

    return {'total': len(graphs), 'graphs': graphs}


@app.get("/api/visualizations/{name}", tags=["Visualizations"])
def get_visualization(name: str):
    """Serve a visualization PNG."""
    filename = f"{name}.png"

    # Check original figures first
    path = os.path.join(FIGURES_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")

    # Check API figures
    path = os.path.join(FIGURES_API_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")

    raise HTTPException(status_code=404, detail=f"Visualization '{name}' not found")


# ============================================================
# Root / Health
# ============================================================

@app.get("/", tags=["Health"])
def root():
    return {
        'name': 'Football Salary Prediction API',
        'version': '1.0.0',
        'status': 'running',
        'players_loaded': len(STATE.get('players_df', [])),
        'docs_url': '/docs',
    }


@app.get("/api/health", tags=["Health"])
def health():
    return {
        'status': 'healthy',
        'models_loaded': 'rf_model' in STATE,
        'players': len(STATE.get('players_df', [])),
        'features': len(STATE.get('feature_names', [])),
    }
