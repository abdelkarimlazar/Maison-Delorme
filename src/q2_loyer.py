from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
FIG = RACINE / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
b = pd.read_csv(RACINE / "data" / "clean" / "boutiques_propres.csv")

# boutiques avec CA et surface connus
d = b.dropna(subset=["ca_mensuel_eur", "surface_m2"]).copy()
print("Boutiques utilisees :", len(d))

# correlations:pearson=lien en ligne droite,spearman=lien de rang 
paires = [("loyer_mensuel_eur", "passage_pietons_jour"),
          ("loyer_mensuel_eur", "ca_mensuel_eur"),
          ("loyer_mensuel_eur", "surface_m2"),
          ("passage_pietons_jour", "ca_mensuel_eur")]
for a, c in paires:
    pearson = d[a].corr(d[c])
    spearman = d[a].corr(d[c], method="spearman")
    print(a, "/", c, "| Pearson", round(pearson, 2), "| Spearman", round(spearman, 2),
          "| R2", round(pearson ** 2, 2))

# loyer, passage et CA moyens par type d'emplacement
print()
print(b.groupby("emplacement")[["loyer_mensuel_eur", "passage_pietons_jour", "ca_mensuel_eur"]].mean().round(0))

# graphiques
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for i, (a, c) in enumerate([("surface_m2", "loyer_mensuel_eur"),
                            ("loyer_mensuel_eur", "passage_pietons_jour"),
                            ("passage_pietons_jour", "ca_mensuel_eur")]):
    ax[i].scatter(d[a], d[c])
    ax[i].set_xlabel(a)
    ax[i].set_ylabel(c)
plt.tight_layout()
plt.savefig(FIG / "q2_loyer.png", dpi=150)
print("\nGraphique enregistre")