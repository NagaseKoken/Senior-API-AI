"""
Save Model Script
=================
Replicates the training pipeline from football_salary_prediction.py,
trains all models, and saves artifacts to models/ for the API.

Run this once before starting the API:
    python scripts/save_model.py
"""

import sys
import os
import json
import sqlite3
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'football_data.db')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# Custom Metrics
# ============================================================

def accuracy_within_pct(y_true_log, y_pred_log, pct=0.30):
    """Calculate % of predictions within pct of actual salary (original EUR scale)."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mask = y_true > 0
    within = np.abs(y_pred[mask] - y_true[mask]) / y_true[mask] <= pct
    return np.mean(within) * 100


def range_accuracy(y_true_log, pred_low_log, pred_high_log, tolerance=0.30):
    """% of actuals within tolerance of the predicted range bounds.

    - If actual is inside [low, high] -> correct
    - If actual < low -> correct if (low - actual) / actual <= tolerance
    - If actual > high -> correct if (actual - high) / actual <= tolerance
    """
    y_true = np.expm1(y_true_log)
    low = np.expm1(pred_low_log)
    high = np.expm1(pred_high_log)

    correct = 0
    for actual, lo, hi in zip(y_true, low, high):
        if lo <= actual <= hi:
            correct += 1
        elif actual < lo:
            if (lo - actual) / actual <= tolerance:
                correct += 1
        else:
            if (actual - hi) / actual <= tolerance:
                correct += 1
    return correct / len(y_true) * 100


def predict_range(rf_model, X, percentile_low=25, percentile_high=75):
    """Get prediction range from RF individual trees."""
    tree_preds = np.array([tree.predict(X) for tree in rf_model.estimators_])
    low = np.percentile(tree_preds, percentile_low, axis=0)
    high = np.percentile(tree_preds, percentile_high, axis=0)
    center = rf_model.predict(X)
    return center, low, high


# ============================================================
# Data Loading & Preprocessing (replicates football_salary_prediction.py)
# ============================================================

print("=" * 60)
print("FOOTBALL SALARY PREDICTION - MODEL SAVING SCRIPT")
print("=" * 60)

print("\n[1/8] Loading data from SQLite database...")
conn = sqlite3.connect(DB_PATH)
salaries = pd.read_sql("SELECT * FROM salaries", conn)
sofifa = pd.read_sql("SELECT * FROM sofifa_attributes", conn)
stats = pd.read_sql("SELECT * FROM player_stats", conn)
market = pd.read_sql("SELECT * FROM market_values", conn)
conn.close()

# Drop duplicate ID columns (same as original script)
sofifa = sofifa.drop(columns=['id', 'player_id', 'short_name', 'long_name', 'player_url',
                               'fifa_version', 'fifa_update', 'fifa_update_date'], errors='ignore')
stats = stats.drop(columns=['id', 'player_name'], errors='ignore')
market = market.drop(columns=['id'], errors='ignore')

# Merge tables (salaries already has short_name, long_name)
df = salaries.merge(sofifa, on='player_pk', how='left', suffixes=('', '_sofifa'))
df = df.merge(stats, on='player_pk', how='left', suffixes=('', '_stats'))
df = df.merge(market, on='player_pk', how='left', suffixes=('', '_market'))

# Filter
df = df[df['gross_annual_eur'] > 0]
df = df[df['appearances'] >= 20]
df = df[df['league_name'].isin(['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1'])]
df.drop(columns=['weekly_wage_eur', 'player_id', 'gross_weekly_eur', 'api_football_id'],
        inplace=True, errors='ignore')

print(f"   Dataset: {df.shape[0]} players, {df.shape[1]} columns")

# Save full player dataframe for API use
player_info = df[['player_pk', 'short_name', 'long_name', 'gross_annual_eur',
                   'position', 'preferred_foot', 'league_name', 'age', 'overall',
                   'potential', 'international_reputation', 'value_eur', 'wage_eur',
                   'market_value_eur', 'appearances', 'goals', 'assists', 'rating',
                   'shooting', 'passing', 'dribbling', 'defending', 'physic',
                   'league_level', 'movement_reactions', 'mentality_composure',
                   'release_clause_eur', 'minutes']].copy()

# ============================================================
# Feature Engineering (replicates lines 160-219)
# ============================================================

print("[2/8] Engineering features...")

sofifa_features = ['overall', 'potential', 'value_eur', 'wage_eur', 'age',
                   'international_reputation', 'shooting', 'passing',
                   'dribbling', 'defending', 'physic', 'league_level',
                   'movement_reactions', 'mentality_composure',
                   'release_clause_eur']
stats_features = ['appearances', 'minutes', 'rating', 'goals', 'assists']
market_features = ['market_value_eur']

feature_cols = sofifa_features + stats_features + market_features
feature_cols = [f for f in feature_cols if f in df.columns]

model_df = df[feature_cols + ['gross_annual_eur', 'position', 'preferred_foot']].copy()
model_df = model_df.reset_index(drop=True)

# Encode preferred_foot
le_foot = LabelEncoder()
model_df['preferred_foot'] = le_foot.fit_transform(model_df['preferred_foot'].fillna('Right'))

# One-hot encode position
position_categories = sorted(model_df['position'].dropna().unique().tolist())
pos_dummies = pd.get_dummies(model_df['position'], prefix='pos', drop_first=True).astype(int)
model_df = pd.concat([model_df, pos_dummies], axis=1)
model_df.drop('position', axis=1, inplace=True)

# Fill numeric NaN with median — save medians for API
feature_medians = {}
for col in model_df.columns:
    if model_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
        med = model_df[col].median()
        feature_medians[col] = float(med)
        model_df[col] = model_df[col].fillna(med)

# Feature engineering — log transforms
model_df['log_value_eur'] = np.log1p(model_df['value_eur'])
model_df['log_wage_eur'] = np.log1p(model_df['wage_eur'])
model_df['log_market_value'] = np.log1p(model_df['market_value_eur'])
model_df['log_release_clause'] = np.log1p(model_df['release_clause_eur']) if 'release_clause_eur' in model_df.columns else 0

# Interaction features
model_df['goals_per_90'] = np.where(model_df['minutes'] > 0,
                                     model_df['goals'] / (model_df['minutes'] / 90), 0)
model_df['assists_per_90'] = np.where(model_df['minutes'] > 0,
                                       model_df['assists'] / (model_df['minutes'] / 90), 0)
model_df['goal_contributions'] = model_df['goals'] + model_df['assists']
model_df['age_squared'] = model_df['age'] ** 2
model_df['overall_x_reputation'] = model_df['overall'] * model_df['international_reputation']
model_df['value_per_overall'] = np.where(model_df['overall'] > 0,
                                          model_df['value_eur'] / model_df['overall'], 0)
model_df['wage_to_value_ratio'] = np.where(model_df['value_eur'] > 0,
                                            model_df['wage_eur'] / model_df['value_eur'], 0)

# Replace inf and remaining NaN
model_df.replace([np.inf, -np.inf], 0, inplace=True)
model_df = model_df.fillna(0)
model_df = model_df.select_dtypes(include=[np.number])

print(f"   Feature matrix: {model_df.shape}")

# ============================================================
# Train/Test Split & Scaling
# ============================================================

print("[3/8] Splitting and scaling data...")

y = np.log1p(model_df['gross_annual_eur'])
X = model_df.drop('gross_annual_eur', axis=1)
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================
# Train Base Models
# ============================================================

print("[4/8] Training base models...")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
}

results = {}
predictions = {}

for name, model in models.items():
    if name in ['Linear Regression', 'Ridge', 'Lasso']:
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    results[name] = {
        'Train R2': round(float(r2_score(y_train, y_pred_train)), 4),
        'Test R2': round(float(r2_score(y_test, y_pred_test)), 4),
        'CV R2 Mean': round(float(cv_scores.mean()), 4),
        'CV R2 Std': round(float(cv_scores.std()), 4),
        'MAE': round(float(mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_test))), 0),
        'RMSE': round(float(np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred_test)))), 0),
        'Within 30%': round(float(accuracy_within_pct(y_test, y_pred_test, 0.30)), 1)
    }
    predictions[name] = y_pred_test
    print(f"   {name}: Test R2={results[name]['Test R2']}, Within 30%={results[name]['Within 30%']}%")

# ============================================================
# Hyperparameter Tuning
# ============================================================

print("[5/8] Tuning models (this may take a few minutes)...")

# Tune Random Forest
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    {'n_estimators': [300, 500, 800], 'max_depth': [6, 8, 10], 'min_samples_leaf': [10, 15, 20]},
    cv=5, scoring='r2', n_jobs=-1
)
rf_grid.fit(X_train, y_train)
print(f"   RF Best CV R2: {rf_grid.best_score_:.4f}, Params: {rf_grid.best_params_}")

# Tune XGBoost
xgb_grid = GridSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    {'n_estimators': [500, 800, 1000], 'max_depth': [3, 4, 5],
     'learning_rate': [0.005, 0.01, 0.05], 'subsample': [0.7, 0.8],
     'colsample_bytree': [0.7, 0.8],
     'reg_alpha': [0.1, 1.0, 5.0], 'reg_lambda': [1.0, 5.0]},
    cv=5, scoring='r2', n_jobs=-1
)
xgb_grid.fit(X_train, y_train)
print(f"   XGB Best CV R2: {xgb_grid.best_score_:.4f}, Params: {xgb_grid.best_params_}")

# Tune Gradient Boosting
gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    {'n_estimators': [500, 800, 1000], 'max_depth': [3, 4],
     'learning_rate': [0.005, 0.01, 0.05], 'subsample': [0.7, 0.8],
     'min_samples_leaf': [10, 15]},
    cv=5, scoring='r2', n_jobs=-1
)
gb_grid.fit(X_train, y_train)
print(f"   GB Best CV R2: {gb_grid.best_score_:.4f}, Params: {gb_grid.best_params_}")

# Stacking Ensemble
stacking = StackingRegressor(
    estimators=[
        ('ridge', Ridge(alpha=1.0)),
        ('lasso', Lasso(alpha=0.01)),
        ('rf', rf_grid.best_estimator_),
        ('gb', gb_grid.best_estimator_),
        ('xgb', xgb_grid.best_estimator_)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5, n_jobs=-1
)
stacking.fit(X_train_scaled, y_train)

# ============================================================
# Evaluate Tuned Models
# ============================================================

print("[6/8] Evaluating tuned models...")

tuned_models = {
    'Tuned RF': (rf_grid.best_estimator_, X_test, X_train, False),
    'Tuned XGBoost': (xgb_grid.best_estimator_, X_test, X_train, False),
    'Tuned GB': (gb_grid.best_estimator_, X_test, X_train, False),
    'Stacking Ensemble': (stacking, X_test_scaled, X_train_scaled, True)
}

best_point_acc = 0
best_model_name = ''
best_model = None
best_pred = None
best_requires_scaling = False

# RF is always used for range predictions (via individual trees)
center_rf_test, low_rf_test, high_rf_test = predict_range(rf_grid.best_estimator_, X_test)
rf_range_acc = range_accuracy(y_test, low_rf_test, high_rf_test, 0.30)

for name, (model, X_eval, X_tr, needs_scaling) in tuned_models.items():
    pred = model.predict(X_eval)
    r2 = r2_score(y_test, pred)
    point_acc = accuracy_within_pct(y_test, pred, 0.30)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(pred))
    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred)))
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')

    results[name] = {
        'Train R2': round(float(r2_score(y_train, model.predict(X_tr))), 4),
        'Test R2': round(float(r2), 4),
        'CV R2 Mean': round(float(cv_scores.mean()), 4),
        'CV R2 Std': round(float(cv_scores.std()), 4),
        'MAE': round(float(mae), 0),
        'RMSE': round(float(rmse), 0),
        'Within 30%': round(float(point_acc), 1),
        'Range Accuracy': round(float(rf_range_acc), 1)
    }
    predictions[name] = pred

    print(f"   {name}: R2={r2:.4f}, Within 30%={point_acc:.1f}%, Range Accuracy={rf_range_acc:.1f}%")

    # Select best model by point accuracy (RF is always used for ranges)
    if point_acc > best_point_acc:
        best_point_acc = point_acc
        best_model_name = name
        best_model = model
        best_pred = pred
        best_requires_scaling = needs_scaling

# The RF model is always used for range predictions
rf_model = rf_grid.best_estimator_

# Compute range predictions for the full dataset
X_all = model_df.drop('gross_annual_eur', axis=1)
center_all, low_all, high_all = predict_range(rf_model, X_all)

# Compute residual std for confidence intervals
residuals_log = y_test.values - best_pred
residual_std = float(np.std(residuals_log))

# Final metrics
y_pred_final = best_pred
final_r2 = r2_score(y_test, y_pred_final)
final_mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_final))
final_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred_final)))
final_point_acc = accuracy_within_pct(y_test, y_pred_final, 0.30)

# Range accuracy for the RF model on test set (already computed above)
final_range_acc = rf_range_acc

# Feature importances
if hasattr(best_model, 'feature_importances_'):
    importances = dict(zip(feature_names, [float(x) for x in best_model.feature_importances_]))
else:
    importances = dict(zip(feature_names, [float(x) for x in xgb_grid.best_estimator_.feature_importances_]))

# RF importances for range model
rf_importances = dict(zip(feature_names, [float(x) for x in rf_model.feature_importances_]))

print(f"\n   Best Model: {best_model_name}")
print(f"   R2: {final_r2:.4f}")
print(f"   MAE: {final_mae:,.0f} EUR")
print(f"   Point Within 30%: {final_point_acc:.1f}%")
print(f"   Range Accuracy (30% tolerance): {final_range_acc:.1f}%")

# ============================================================
# Compute All Player Predictions
# ============================================================

print("[7/8] Computing predictions for all players...")

# Prepare player predictions dataframe
player_info = player_info.reset_index(drop=True)
player_info['predicted_center_eur'] = np.expm1(center_all)
player_info['predicted_low_eur'] = np.expm1(low_all)
player_info['predicted_high_eur'] = np.expm1(high_all)
player_info['prediction_error_pct'] = np.where(
    player_info['gross_annual_eur'] > 0,
    (player_info['predicted_center_eur'] - player_info['gross_annual_eur']) / player_info['gross_annual_eur'] * 100,
    0
)

# Determine if actual is in range
actual = player_info['gross_annual_eur'].values
low_eur = player_info['predicted_low_eur'].values
high_eur = player_info['predicted_high_eur'].values
in_range = (actual >= low_eur) & (actual <= high_eur)
player_info['actual_in_range'] = in_range

# Range accuracy result (within 30% of nearest bound)
range_correct = []
for a, lo, hi in zip(actual, low_eur, high_eur):
    if lo <= a <= hi:
        range_correct.append(True)
    elif a < lo:
        range_correct.append((lo - a) / a <= 0.30)
    else:
        range_correct.append((a - hi) / a <= 0.30)
player_info['range_accuracy_result'] = range_correct

# Train/test split indices
train_indices = list(X_train.index)
test_indices = list(X_test.index)
player_info['split'] = 'train'
player_info.loc[player_info.index.isin(test_indices), 'split'] = 'test'

print(f"   All {len(player_info)} players predicted")
print(f"   Overall range accuracy: {player_info['range_accuracy_result'].mean() * 100:.1f}%")

# ============================================================
# Save Artifacts
# ============================================================

print("[8/8] Saving artifacts to models/...")

# Models
joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.joblib'))
joblib.dump(rf_model, os.path.join(MODELS_DIR, 'rf_model.joblib'))
joblib.dump({
    **{name: model for name, model in models.items()},
    'Tuned RF': rf_grid.best_estimator_,
    'Tuned XGBoost': xgb_grid.best_estimator_,
    'Tuned GB': gb_grid.best_estimator_,
    'Stacking Ensemble': stacking
}, os.path.join(MODELS_DIR, 'all_models.joblib'))

# Preprocessing artifacts
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.joblib'))
joblib.dump(le_foot, os.path.join(MODELS_DIR, 'label_encoder_foot.joblib'))
joblib.dump(feature_medians, os.path.join(MODELS_DIR, 'feature_medians.joblib'))
joblib.dump(position_categories, os.path.join(MODELS_DIR, 'position_categories.joblib'))

# Player predictions
joblib.dump(player_info, os.path.join(MODELS_DIR, 'all_predictions.joblib'))

# Feature matrix for similar players
joblib.dump(X_all.values, os.path.join(MODELS_DIR, 'feature_matrix.joblib'))

# Model config
model_config = {
    'best_model_name': best_model_name,
    'requires_scaling': best_requires_scaling,
    'residual_std': residual_std,
    'n_features': len(feature_names),
    'n_players': len(player_info),
    'test_size': len(X_test),
    'train_size': len(X_train),
    'rf_best_params': rf_grid.best_params_,
    'xgb_best_params': {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in xgb_grid.best_params_.items()},
    'gb_best_params': {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in gb_grid.best_params_.items()},
}
with open(os.path.join(MODELS_DIR, 'model_config.json'), 'w') as f:
    json.dump(model_config, f, indent=2)

# Model metrics
with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Feature importances
with open(os.path.join(MODELS_DIR, 'feature_importances.json'), 'w') as f:
    json.dump({
        'best_model': importances,
        'rf_model': rf_importances
    }, f, indent=2)

# Dataset stats
dataset_stats = {
    'total_players': int(len(player_info)),
    'salary_mean': float(player_info['gross_annual_eur'].mean()),
    'salary_median': float(player_info['gross_annual_eur'].median()),
    'salary_min': float(player_info['gross_annual_eur'].min()),
    'salary_max': float(player_info['gross_annual_eur'].max()),
    'salary_std': float(player_info['gross_annual_eur'].std()),
    'leagues': {},
    'positions': {},
    'age_stats': {
        'mean': float(player_info['age'].mean()),
        'min': int(player_info['age'].min()),
        'max': int(player_info['age'].max()),
    },
    'final_metrics': {
        'best_model': best_model_name,
        'r2': round(float(final_r2), 4),
        'mae': round(float(final_mae), 0),
        'rmse': round(float(final_rmse), 0),
        'point_within_30': round(float(final_point_acc), 1),
        'range_accuracy_30': round(float(final_range_acc), 1),
    }
}

for league in player_info['league_name'].unique():
    ldf = player_info[player_info['league_name'] == league]
    dataset_stats['leagues'][league] = {
        'count': int(len(ldf)),
        'mean_salary': float(ldf['gross_annual_eur'].mean()),
        'median_salary': float(ldf['gross_annual_eur'].median()),
        'min_salary': float(ldf['gross_annual_eur'].min()),
        'max_salary': float(ldf['gross_annual_eur'].max()),
    }

for pos in player_info['position'].dropna().unique():
    pdf = player_info[player_info['position'] == pos]
    dataset_stats['positions'][pos] = {
        'count': int(len(pdf)),
        'mean_salary': float(pdf['gross_annual_eur'].mean()),
        'median_salary': float(pdf['gross_annual_eur'].median()),
    }

with open(os.path.join(MODELS_DIR, 'dataset_stats.json'), 'w') as f:
    json.dump(dataset_stats, f, indent=2)

print("\n" + "=" * 60)
print("ALL ARTIFACTS SAVED SUCCESSFULLY")
print("=" * 60)
print(f"\nSaved to: {MODELS_DIR}")
print(f"Files: {os.listdir(MODELS_DIR)}")
print(f"\nBest Model: {best_model_name}")
print(f"Range Accuracy (30% tolerance): {final_range_acc:.1f}%")
print(f"Point Within 30%: {final_point_acc:.1f}%")
print(f"Improvement: +{final_range_acc - final_point_acc:.1f} percentage points")
