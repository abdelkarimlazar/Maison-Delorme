from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RACINE = Path(__file__).resolve().parents[1]
DATA_PATH = RACINE / "data" / "clean" / "boutiques_propres.csv"
OUTPUT_DIR = RACINE / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = ["surface_m2", "passage_pietons_jour", "concurrents_300m"]
target = "ca_mensuel_eur"

data = df.dropna(subset=[target] + features).copy()
X = data[features]
y = data[target]

print("Boutiques totales :", len(df))
print("Boutiques utilisables (CA et variables connus) :", len(data))

# Validation croisée 5-fold pour comparer les modèles sans fuite de données
cv = KFold(n_splits=5, shuffle=True, random_state=42)

modeles = {
    "Régression linéaire": LinearRegression(),
    "Ridge (alpha=1.0)": Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42),
}

print("\n=== Q4 : COMPARAISON DES MODELES (Validation croisée 5-fold) ===")
resultats = {}
for nom, modele in modeles.items():
    y_pred_cv = cross_val_predict(modele, X, y, cv=cv)
    r2 = r2_score(y, y_pred_cv)
    mae = mean_absolute_error(y, y_pred_cv)
    rmse = root_mean_squared_error(y, y_pred_cv)
    mape = mean_absolute_percentage_error(y, y_pred_cv) * 100
    resultats[nom] = {"y_pred_cv": y_pred_cv, "r2": r2, "mae": mae, "rmse": rmse, "mape": mape}
    print(f"{nom:22s} | R2 = {r2:.3f} | MAE = {mae:,.0f} € | RMSE = {rmse:,.0f} € | MAPE = {mape:.1f} %")

# Modèle retenu : Régression linéaire (meilleur R2/erreur en CV + interprétabilité directe)
reg = LinearRegression()
reg.fit(X, y)

# Coefficients standardisés pour comparer l'importance relative des variables
reg_std = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
reg_std.fit(X, y)
coefs_std = reg_std.named_steps["reg"].coef_

print("\n=== Q4 : EXPLICATION DU MODELE RETENU (Régression linéaire) ===")
print(f"Ordonnée à l'origine (intercept) : {reg.intercept_:,.2f} €")
for f, c, cs in zip(features, reg.coef_, coefs_std):
    print(f"  {f:22s} : {c:+8.2f} € / unité  (coef. standardisé = {cs:+,.0f} €)")

# Prédiction pour le local proposé en Q3
local_propose = pd.DataFrame([{
    "surface_m2": 95,
    "passage_pietons_jour": 4200,
    "concurrents_300m": 2,
}])
ca_estime = reg.predict(local_propose)[0]
mae_cv = resultats["Régression linéaire"]["mae"]
rmse_cv = resultats["Régression linéaire"]["rmse"]
mape_cv = resultats["Régression linéaire"]["mape"]
r2_cv = resultats["Régression linéaire"]["r2"]

print("\n=== Q3 : ESTIMATION POUR LE LOCAL PROPOSE ===")
print("Local : 95 m², 4 200 passages/jour, 2 concurrents < 300 m")
print(f"CA mensuel estimé : {ca_estime:,.0f} €")
print(f"Fourchette indicative (± MAE) : [{ca_estime - mae_cv:,.0f} € ; {ca_estime + mae_cv:,.0f} €]")

# Graphiques de diagnostic (Q3-Q4)
y_pred_cv = resultats["Régression linéaire"]["y_pred_cv"]
residus = y - y_pred_cv

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1. Prédit vs Réel
axes[0].scatter(y, y_pred_cv, alpha=0.75, edgecolors="k", linewidths=0.5, label="Boutiques (CV 5-fold)")
lim_min = min(y.min(), y_pred_cv.min()) * 0.9
lim_max = max(y.max(), y_pred_cv.max()) * 1.05
axes[0].plot([lim_min, lim_max], [lim_min, lim_max], "r--", label="Prédiction parfaite (y = x)")
axes[0].axhline(ca_estime, color="green", linestyle=":", linewidth=1.5, label=f"Local Q3 estimé ({ca_estime:,.0f} €)")
axes[0].set_xlabel("CA mensuel réel (€)")
axes[0].set_ylabel("CA mensuel prédit (€)")
axes[0].set_title(f"Q3-Q4 : Réel vs Prédit (R² CV = {r2_cv:.2f}, MAPE = {mape_cv:.1f}%)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Résidus vs Prédit
axes[1].scatter(y_pred_cv, residus, alpha=0.75, edgecolors="k", linewidths=0.5)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].axhline(mae_cv, color="orange", linestyle=":", label=f"± MAE ({mae_cv:,.0f} €)")
axes[1].axhline(-mae_cv, color="orange", linestyle=":")
axes[1].set_xlabel("CA mensuel prédit (€)")
axes[1].set_ylabel("Résidu = Réel - Prédit (€)")
axes[1].set_title("Diagnostic des résidus (Validation croisée)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "q3_q4_regression.png", dpi=150)
plt.close()
print("\nGraphique enregistré dans", FIG_DIR / "q3_q4_regression.png")

# Sauvegarde des indicateurs texte
with open(OUTPUT_DIR / "q3_q4_indicateurs.txt", "w", encoding="utf-8") as f:
    f.write("Q3-Q4 : Régression linéaire (Validation croisée 5-fold)\n")
    f.write(f"Nombre de boutiques : {len(data)}\n")
    f.write(f"R2 (CV) : {r2_cv:.3f}\n")
    f.write(f"MAE (CV) : {mae_cv:.0f} EUR\n")
    f.write(f"RMSE (CV) : {rmse_cv:.0f} EUR\n")
    f.write(f"MAPE (CV) : {mape_cv:.1f} %\n\n")
    f.write("Coefficients (modèle ajusté sur les 56 boutiques) :\n")
    f.write(f"  Intercept : {reg.intercept_:.2f} EUR\n")
    for feat, c, cs in zip(features, reg.coef_, coefs_std):
        f.write(f"  {feat} : {c:+.2f} EUR/unité (std: {cs:+.0f} EUR)\n")
    f.write(f"\nLocal proposé (95 m2, 4200 passages/j, 2 concurrents) : {ca_estime:.0f} EUR\n")
    f.write(f"Fourchette (± MAE) : [{ca_estime - mae_cv:.0f} ; {ca_estime + mae_cv:.0f}] EUR\n")

print("Indicateurs enregistrés dans", OUTPUT_DIR / "q3_q4_indicateurs.txt")
