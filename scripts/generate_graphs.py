"""
Generate Academic Graphs
========================
Generates additional visualizations for the academic report.
Saves all figures to figures_api/ directory.

Run after save_model.py:
    python scripts/generate_graphs.py
"""

import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures_api')
DB_PATH = os.path.join(BASE_DIR, 'data', 'football_data.db')
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, f'{name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   Saved: {name}.png")


# ============================================================
# Load Data
# ============================================================

print("Loading model artifacts...")
rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.joblib'))
feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))
feature_matrix = joblib.load(os.path.join(MODELS_DIR, 'feature_matrix.joblib'))
players_df = joblib.load(os.path.join(MODELS_DIR, 'all_predictions.joblib'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))

with open(os.path.join(MODELS_DIR, 'model_metrics.json')) as f:
    model_metrics = json.load(f)
with open(os.path.join(MODELS_DIR, 'model_config.json')) as f:
    model_config = json.load(f)
with open(os.path.join(MODELS_DIR, 'feature_importances.json')) as f:
    feature_importances = json.load(f)

# Load full data for some plots
conn = sqlite3.connect(DB_PATH)
salaries = pd.read_sql("SELECT * FROM salaries", conn)
sofifa = pd.read_sql("SELECT * FROM sofifa_attributes", conn)
stats = pd.read_sql("SELECT * FROM player_stats", conn)
market = pd.read_sql("SELECT * FROM market_values", conn)
conn.close()

sofifa = sofifa.drop(columns=['id', 'player_id', 'short_name', 'long_name', 'player_url',
                               'fifa_version', 'fifa_update', 'fifa_update_date'], errors='ignore')
stats = stats.drop(columns=['id', 'player_name'], errors='ignore')
market = market.drop(columns=['id'], errors='ignore')

full_df = salaries.merge(sofifa, on='player_pk', how='left', suffixes=('', '_sofifa'))
full_df = full_df.merge(stats, on='player_pk', how='left', suffixes=('', '_stats'))
full_df = full_df.merge(market, on='player_pk', how='left', suffixes=('', '_market'))
full_df = full_df[full_df['gross_annual_eur'] > 0]
full_df = full_df[full_df['appearances'] >= 20]
full_df = full_df[full_df['league_name'].isin(['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1'])]

# Prepare X and y for SHAP and learning curves
from sklearn.model_selection import train_test_split
y_all = np.log1p(players_df['gross_annual_eur'].values)
X_all = pd.DataFrame(feature_matrix, columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

print(f"Loaded {len(players_df)} players, {len(feature_names)} features")
print(f"\nGenerating graphs...")

# ============================================================
# 1. SHAP Summary Plot
# ============================================================

print("\n[1/16] SHAP Summary Plot...")
try:
    import shap
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title('SHAP Feature Importance (Beeswarm Plot)', fontsize=14, pad=20)
    plt.tight_layout()
    save_fig(plt.gcf(), 'shap_summary')
except Exception as e:
    print(f"   SHAP summary failed: {e}")

# ============================================================
# 2-5. SHAP Dependence Plots
# ============================================================

shap_features = {
    'overall': 'Overall Rating',
    'log_value_eur': 'Log(FIFA Value)',
    'log_wage_eur': 'Log(Wage)',
    'log_market_value': 'Log(Market Value)',
}

for i, (feat, title) in enumerate(shap_features.items(), 2):
    print(f"[{i}/16] SHAP Dependence: {feat}...")
    try:
        feat_idx = feature_names.index(feat)
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.dependence_plot(feat_idx, shap_values, X_test,
                            feature_names=feature_names, show=False, ax=ax)
        ax.set_title(f'SHAP Dependence Plot: {title}', fontsize=14)
        plt.tight_layout()
        save_fig(fig, f'shap_dependence_{feat.replace("log_", "")}')
    except Exception as e:
        print(f"   SHAP dependence {feat} failed: {e}")

# ============================================================
# 6. Learning Curves
# ============================================================

print("[6/16] Learning Curves...")
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    rf_model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='r2', n_jobs=-1
)

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training Score', linewidth=2)
ax.fill_between(train_sizes,
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation Score', linewidth=2)
ax.fill_between(train_sizes,
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15)
ax.set_xlabel('Training Set Size', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('Learning Curves (Random Forest)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig(fig, 'learning_curves')

# ============================================================
# 7. Prediction Error Distribution
# ============================================================

print("[7/16] Prediction Error Distribution...")
pred_center = players_df['predicted_center_eur'].values
actual = players_df['gross_annual_eur'].values
error_pct = (pred_center - actual) / actual * 100

fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(error_pct, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(x=-30, color='red', linestyle='--', linewidth=2, label='-30% threshold')
ax.axvline(x=30, color='red', linestyle='--', linewidth=2, label='+30% threshold')
ax.axvline(x=0, color='green', linestyle='-', linewidth=2, label='Perfect prediction')

within_30 = np.mean(np.abs(error_pct) <= 30) * 100
ax.text(0.02, 0.95, f'Within ±30%: {within_30:.1f}%',
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('Prediction Error (%)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Prediction Error Distribution (Center Point)', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
save_fig(fig, 'prediction_error_distribution')

# ============================================================
# 8. Cross-Validation Box Plots
# ============================================================

print("[8/16] Cross-Validation Box Plots...")
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

cv_models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'RF (Tuned)': rf_model,
    'GB': GradientBoostingRegressor(random_state=42, **model_config.get('gb_best_params', {})),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0, **{k: v for k, v in model_config.get('xgb_best_params', {}).items()}),
}

cv_results = {}
scaler_obj = scaler
X_train_scaled = scaler_obj.transform(X_train)

for name, model in cv_models.items():
    if name in ['Ridge', 'Lasso']:
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_results[name] = scores

fig, ax = plt.subplots(figsize=(10, 7))
bp = ax.boxplot(cv_results.values(), labels=cv_results.keys(), patch_artist=True)
colors = sns.color_palette('husl', len(cv_results))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('5-Fold Cross-Validation R² Scores', fontsize=14)
ax.tick_params(axis='x', rotation=15)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig(fig, 'cv_boxplots')

# ============================================================
# 9. Cumulative Accuracy Curve
# ============================================================

print("[9/16] Cumulative Accuracy Curve...")
abs_error_pct = np.abs(error_pct)
thresholds = np.arange(0, 101, 1)
cum_acc = [np.mean(abs_error_pct <= t) * 100 for t in thresholds]

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(thresholds, cum_acc, linewidth=2.5, color='steelblue')
ax.axvline(x=30, color='red', linestyle='--', linewidth=1.5, label='30% threshold')
ax.axhline(y=cum_acc[30], color='red', linestyle=':', alpha=0.5)
ax.scatter([10, 20, 30, 50], [cum_acc[10], cum_acc[20], cum_acc[30], cum_acc[50]],
           s=80, zorder=5, color='red')
for t in [10, 20, 30, 50]:
    ax.annotate(f'{t}%: {cum_acc[t]:.1f}%', (t, cum_acc[t]),
                textcoords="offset points", xytext=(10, -15), fontsize=10)

ax.set_xlabel('Error Threshold (%)', fontsize=12)
ax.set_ylabel('% of Predictions Within Threshold', fontsize=12)
ax.set_title('Cumulative Prediction Accuracy Curve', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)
plt.tight_layout()
save_fig(fig, 'cumulative_accuracy')

# ============================================================
# 10. Model Complexity vs Performance
# ============================================================

print("[10/16] Complexity vs Performance...")
complexity = {
    'Ridge': 36, 'Lasso': 36,
    'Random Forest': 200 * 15, 'Gradient Boosting': 200 * 5,
    'XGBoost': 200 * 5,
    'Tuned RF': model_config['rf_best_params']['n_estimators'] * model_config['rf_best_params']['max_depth'],
    'Tuned XGBoost': model_config['xgb_best_params']['n_estimators'] * model_config['xgb_best_params']['max_depth'],
    'Tuned GB': model_config['gb_best_params']['n_estimators'] * model_config['gb_best_params']['max_depth'],
}
r2_values = {name: model_metrics[name]['Test R2'] for name in complexity if name in model_metrics}
within30 = {name: model_metrics[name]['Within 30%'] for name in complexity if name in model_metrics}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

names = list(r2_values.keys())
x = [complexity[n] for n in names]
y1 = [r2_values[n] for n in names]
y2 = [within30[n] for n in names]

ax1.scatter(x, y1, s=100, zorder=5)
for i, name in enumerate(names):
    ax1.annotate(name, (x[i], y1[i]), textcoords="offset points",
                 xytext=(5, 5), fontsize=9)
ax1.set_xlabel('Model Complexity (estimators × depth)', fontsize=12)
ax1.set_ylabel('Test R²', fontsize=12)
ax1.set_title('Model Complexity vs R² Score', fontsize=14)
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)

ax2.scatter(x, y2, s=100, zorder=5, color='orange')
for i, name in enumerate(names):
    ax2.annotate(name, (x[i], y2[i]), textcoords="offset points",
                 xytext=(5, 5), fontsize=9)
ax2.set_xlabel('Model Complexity (estimators × depth)', fontsize=12)
ax2.set_ylabel('Within 30% Accuracy (%)', fontsize=12)
ax2.set_title('Model Complexity vs Accuracy', fontsize=14)
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_fig(fig, 'complexity_vs_performance')

# ============================================================
# 11. League Salary Violin Plots
# ============================================================

print("[11/16] League Salary Violin Plots...")
fig, ax = plt.subplots(figsize=(12, 7))
league_order = full_df.groupby('league_name')['gross_annual_eur'].median().sort_values(ascending=False).index
plot_df = full_df[full_df['league_name'].isin(league_order)].copy()
plot_df['salary_millions'] = plot_df['gross_annual_eur'] / 1e6

sns.violinplot(data=plot_df, x='league_name', y='salary_millions',
               order=league_order, ax=ax, inner='box', cut=0)
ax.set_xlabel('')
ax.set_ylabel('Annual Salary (Millions EUR)', fontsize=12)
ax.set_title('Salary Distribution by League (Violin Plot)', fontsize=14)
ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
save_fig(fig, 'league_salary_violin')

# ============================================================
# 12. Salary vs Age Curves by Position
# ============================================================

print("[12/16] Salary vs Age Curves...")
fig, ax = plt.subplots(figsize=(12, 7))
positions = ['Attacker', 'Midfielder', 'Defender', 'Goalkeeper']
colors = sns.color_palette('husl', 4)

for pos, color in zip(positions, colors):
    pos_df = full_df[full_df['position'] == pos]
    age_salary = pos_df.groupby('age')['gross_annual_eur'].mean() / 1e6
    # Smooth with rolling average
    if len(age_salary) >= 3:
        smoothed = age_salary.rolling(window=3, center=True, min_periods=1).mean()
        ax.plot(smoothed.index, smoothed.values, '-o', label=pos, color=color,
                linewidth=2, markersize=4, alpha=0.8)

ax.scatter(full_df['age'], full_df['gross_annual_eur'] / 1e6,
           alpha=0.08, s=8, color='gray')
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Average Annual Salary (Millions EUR)', fontsize=12)
ax.set_title('Salary vs Age by Position', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(18, 38)
plt.tight_layout()
save_fig(fig, 'salary_age_curves')

# ============================================================
# 13. Position Radar Chart
# ============================================================

print("[13/16] Position Radar Chart...")
radar_attrs = ['shooting', 'passing', 'dribbling', 'defending', 'physic', 'overall']
radar_labels = ['Shooting', 'Passing', 'Dribbling', 'Defending', 'Physical', 'Overall']

fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(polar=True))
positions = ['Attacker', 'Midfielder', 'Defender', 'Goalkeeper']
colors = sns.color_palette('husl', 4)

for ax, pos, color in zip(axes.flat, positions, colors):
    pos_df = full_df[full_df['position'] == pos]
    values = [pos_df[attr].mean() for attr in radar_attrs]
    values += values[:1]  # close the polygon

    angles = np.linspace(0, 2 * np.pi, len(radar_attrs), endpoint=False).tolist()
    angles += angles[:1]

    ax.fill(angles, values, alpha=0.25, color=color)
    ax.plot(angles, values, 'o-', linewidth=2, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title(f'{pos}\n(n={len(pos_df)})', fontsize=13, pad=15)

plt.suptitle('Average Player Attributes by Position', fontsize=16, y=1.02)
plt.tight_layout()
save_fig(fig, 'position_radar')

# ============================================================
# 14. Overpaid/Underpaid Scatter
# ============================================================

print("[14/16] Overpaid/Underpaid Scatter...")
fig, ax = plt.subplots(figsize=(12, 10))
actual_m = players_df['gross_annual_eur'] / 1e6
center_m = players_df['predicted_center_eur'] / 1e6
low_m = players_df['predicted_low_eur'] / 1e6
high_m = players_df['predicted_high_eur'] / 1e6

# Color by overpaid (red) / underpaid (blue) / in range (green)
colors_arr = []
for _, row in players_df.iterrows():
    a = row['gross_annual_eur']
    lo = row['predicted_low_eur']
    hi = row['predicted_high_eur']
    if a > hi:
        colors_arr.append('#e74c3c')  # red - overpaid
    elif a < lo:
        colors_arr.append('#3498db')  # blue - underpaid
    else:
        colors_arr.append('#2ecc71')  # green - in range

ax.scatter(actual_m, center_m, c=colors_arr, alpha=0.5, s=20)

max_val = max(actual_m.max(), center_m.max())
ax.plot([0, max_val], [0, max_val], 'k--', lw=1.5, alpha=0.5, label='Perfect prediction')

# Label extreme cases
extremes = players_df.nlargest(5, 'prediction_error_pct')
extremes = pd.concat([extremes, players_df.nsmallest(5, 'prediction_error_pct')])
for _, row in extremes.iterrows():
    ax.annotate(str(row['short_name']),
                (row['gross_annual_eur'] / 1e6, row['predicted_center_eur'] / 1e6),
                fontsize=7, alpha=0.8,
                textcoords="offset points", xytext=(5, 5))

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Overpaid (above range)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='Underpaid (below range)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, label='In predicted range'),
    Line2D([0], [0], color='black', linestyle='--', label='Perfect prediction'),
]
ax.legend(handles=legend_elements, fontsize=11, loc='upper left')

in_range_pct = players_df['actual_in_range'].mean() * 100
ax.text(0.98, 0.02, f'In range: {in_range_pct:.1f}%',
        transform=ax.transAxes, fontsize=12, ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('Actual Salary (Millions EUR)', fontsize=12)
ax.set_ylabel('Predicted Center Salary (Millions EUR)', fontsize=12)
ax.set_title('Overpaid vs Underpaid Players', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig(fig, 'overpaid_underpaid_scatter')

# ============================================================
# 15. Residuals by League and Position
# ============================================================

print("[15/16] Residuals by League and Position...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

res_df = players_df.copy()
res_df['residual_pct'] = (res_df['predicted_center_eur'] - res_df['gross_annual_eur']) / res_df['gross_annual_eur'] * 100

# By League
league_order = res_df.groupby('league_name')['residual_pct'].median().sort_values().index
sns.boxplot(data=res_df, x='league_name', y='residual_pct', order=league_order, ax=ax1)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
ax1.set_xlabel('')
ax1.set_ylabel('Residual (%)', fontsize=12)
ax1.set_title('Prediction Residuals by League', fontsize=14)
ax1.tick_params(axis='x', rotation=20)

# By Position
pos_order = res_df.groupby('position')['residual_pct'].median().sort_values().index
sns.boxplot(data=res_df, x='position', y='residual_pct', order=pos_order, ax=ax2)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
ax2.set_xlabel('')
ax2.set_ylabel('Residual (%)', fontsize=12)
ax2.set_title('Prediction Residuals by Position', fontsize=14)

plt.tight_layout()
save_fig(fig, 'residuals_by_league_position')

# ============================================================
# 16. Range Prediction Plot
# ============================================================

print("[16/16] Range Prediction Plot...")
fig, ax = plt.subplots(figsize=(14, 8))

# Sort by actual salary and take a sample for readability
sample_df = players_df.sort_values('gross_annual_eur').reset_index(drop=True)
# Take every nth player for visibility
n = max(1, len(sample_df) // 80)
sample_df = sample_df.iloc[::n].reset_index(drop=True)

x = np.arange(len(sample_df))
actual_vals = sample_df['gross_annual_eur'] / 1e6
low_vals = sample_df['predicted_low_eur'] / 1e6
high_vals = sample_df['predicted_high_eur'] / 1e6
center_vals = sample_df['predicted_center_eur'] / 1e6

# Error bars (range)
yerr_low = center_vals - low_vals
yerr_high = high_vals - center_vals

# Color by whether actual is in range
colors_range = ['#2ecc71' if row['actual_in_range'] else '#e74c3c'
                for _, row in sample_df.iterrows()]

ax.errorbar(x, center_vals, yerr=[yerr_low, yerr_high],
            fmt='none', ecolor='lightblue', elinewidth=2, capsize=3, alpha=0.6)
ax.scatter(x, actual_vals, c=colors_range, s=25, zorder=5, alpha=0.7)
ax.scatter(x, center_vals, c='steelblue', s=10, zorder=4, alpha=0.4, marker='_')

in_range_count = sample_df['actual_in_range'].sum()
total = len(sample_df)
ax.text(0.02, 0.95, f'In range: {in_range_count}/{total} ({in_range_count/total*100:.1f}%)',
        transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=8, label='Actual in range'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Actual outside range'),
    Line2D([0], [0], color='lightblue', linewidth=3, label='Predicted range'),
]
ax.legend(handles=legend_elements, fontsize=11)

ax.set_xlabel('Players (sorted by actual salary)', fontsize=12)
ax.set_ylabel('Salary (Millions EUR)', fontsize=12)
ax.set_title('Predicted Salary Ranges vs Actual Salaries', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig(fig, 'range_prediction_plot')

# ============================================================
# Done
# ============================================================

print(f"\n{'='*60}")
print(f"ALL {len(os.listdir(FIGURES_DIR))} GRAPHS GENERATED SUCCESSFULLY")
print(f"{'='*60}")
print(f"Saved to: {FIGURES_DIR}")
print(f"Files: {sorted(os.listdir(FIGURES_DIR))}")
