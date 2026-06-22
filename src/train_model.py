import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "lahore_housing.csv")
MODELS_DIR   = os.path.join(BASE, "models")
OUTPUTS_DIR  = os.path.join(BASE, "outputs")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  Lahore Housing Price Prediction — Training Pipeline")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. EDA Plots ──────────────────────────────────────────────────────────────
print("\n📊 Generating EDA plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Lahore Housing Dataset — Exploratory Data Analysis", fontsize=16, fontweight="bold")

# Price distribution
axes[0,0].hist(df["price_lakhs"], bins=50, color="#2563eb", edgecolor="white", alpha=0.85)
axes[0,0].set_title("Price Distribution (PKR Lakhs)")
axes[0,0].set_xlabel("Price (Lakhs)")
axes[0,0].set_ylabel("Count")

# Price by area (top 10 areas)
area_median = df.groupby("area")["price_lakhs"].median().sort_values(ascending=False).head(10)
axes[0,1].barh(area_median.index, area_median.values, color="#059669")
axes[0,1].set_title("Median Price by Area (Top 10)")
axes[0,1].set_xlabel("Median Price (Lakhs)")

# Size vs Price
axes[0,2].scatter(df["size_marla"], df["price_lakhs"], alpha=0.3, c="#7c3aed", s=10)
axes[0,2].set_title("Size vs Price")
axes[0,2].set_xlabel("Size (Marla)")
axes[0,2].set_ylabel("Price (Lakhs)")

# Bedrooms vs Price
df.boxplot(column="price_lakhs", by="bedrooms", ax=axes[1,0])
axes[1,0].set_title("Price by Bedrooms")
axes[1,0].set_xlabel("Bedrooms")
axes[1,0].set_ylabel("Price (Lakhs)")

# Property type
type_median = df.groupby("property_type")["price_lakhs"].median()
axes[1,1].bar(type_median.index, type_median.values, color=["#f59e0b","#ef4444","#10b981","#3b82f6"])
axes[1,1].set_title("Median Price by Property Type")
axes[1,1].set_ylabel("Median Price (Lakhs)")

# Correlation heatmap (numeric only)
num_cols = ["size_marla","bedrooms","bathrooms","property_age",
            "has_garage","has_servant_quarter","has_basement",
            "corner_plot","near_park","gated_society","price_lakhs"]
corr = df[num_cols].corr()
sns.heatmap(corr, ax=axes[1,2], annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, annot_kws={"size": 7})
axes[1,2].set_title("Feature Correlation")

plt.tight_layout()
eda_path = os.path.join(OUTPUTS_DIR, "eda_plots.png")
plt.savefig(eda_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved → {eda_path}")

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
FEATURES = ["area","size_marla","bedrooms","bathrooms","property_age",
            "has_garage","has_servant_quarter","has_basement",
            "corner_plot","near_park","gated_society","property_type"]
TARGET = "price_lakhs"

X = df[FEATURES]
y = df[TARGET]

CAT_COLS = ["area", "property_type"]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n📂 Split → Train: {len(X_train)}, Test: {len(X_test)}")

# ── 4. Model Training ─────────────────────────────────────────────────────────
MODELS = {
    "Linear Regression":      LinearRegression(),
    "Ridge Regression":       Ridge(alpha=10),
    "Random Forest":          RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting":      GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    "XGBoost":                XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                                           subsample=0.8, colsample_bytree=0.8,
                                           random_state=42, verbosity=0),
}

print("\n🤖 Training models...\n")
results = {}

for name, model in MODELS.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results[name] = {"pipeline": pipe, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
    print(f"   {name:<25} R²={r2:.4f}  MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%")

# ── 5. Best Model ─────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["R2"])
best      = results[best_name]

print(f"\n🏆 Best Model: {best_name}")
print(f"   R²={best['R2']:.4f}  MAE={best['MAE']:.1f} Lakhs  MAPE={best['MAPE']:.1f}%")

# Save best model
model_path = os.path.join(MODELS_DIR, "best_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best["pipeline"], f)
print(f"   Saved → {model_path}")

# ── 6. Comparison Plot ────────────────────────────────────────────────────────
print("\n📈 Generating model comparison & residual plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Evaluation", fontsize=15, fontweight="bold")

names = list(results.keys())
r2s   = [results[n]["R2"]   for n in names]
maes  = [results[n]["MAE"]  for n in names]
mapes = [results[n]["MAPE"] for n in names]

colors = ["#2563eb" if n != best_name else "#059669" for n in names]

axes[0].barh(names, r2s,   color=colors); axes[0].set_title("R² Score (higher = better)"); axes[0].set_xlim(0,1)
axes[1].barh(names, maes,  color=colors); axes[1].set_title("MAE — Lakhs (lower = better)")
axes[2].barh(names, mapes, color=colors); axes[2].set_title("MAPE % (lower = better)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# Predicted vs Actual for best model
y_pred_best = best["pipeline"].predict(X_test)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"{best_name} — Predicted vs Actual", fontsize=13, fontweight="bold")

axes[0].scatter(y_test, y_pred_best, alpha=0.4, c="#2563eb", s=15)
lo, hi = y_test.min(), y_test.max()
axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
axes[0].set_xlabel("Actual Price (Lakhs)"); axes[0].set_ylabel("Predicted Price (Lakhs)")
axes[0].set_title("Predicted vs Actual"); axes[0].legend()

residuals = y_test - y_pred_best
axes[1].scatter(y_pred_best, residuals, alpha=0.4, c="#7c3aed", s=15)
axes[1].axhline(0, color="red", linestyle="--", lw=1.5)
axes[1].set_xlabel("Predicted Price"); axes[1].set_ylabel("Residual")
axes[1].set_title("Residual Plot")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "best_model_eval.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 7. Feature Importance (if tree-based) ─────────────────────────────────────
try:
    inner_model = best["pipeline"].named_steps["model"]
    if hasattr(inner_model, "feature_importances_"):
        prep_out  = best["pipeline"].named_steps["prep"]
        num_names = NUM_COLS
        cat_names = list(prep_out.named_transformers_["cat"].get_feature_names_out(CAT_COLS))
        feat_names = num_names + cat_names

        importances = inner_model.feature_importances_
        idx = np.argsort(importances)[-20:]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh([feat_names[i] for i in idx], importances[idx], color="#059669")
        ax.set_title(f"Top 20 Feature Importances — {best_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("   Feature importance plot saved.")
except Exception as e:
    print(f"   (Skipped feature importance: {e})")

# ── 8. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅ Training Complete!")
print(f"  Best model : {best_name}")
print(f"  R² Score   : {best['R2']:.4f}")
print(f"  MAE        : {best['MAE']:.1f} Lakhs")
print(f"  MAPE       : {best['MAPE']:.1f}%")
print(f"  Model saved: {model_path}")
print("=" * 60)
