from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RACINE = Path(__file__).resolve().parents[1]
b = pd.read_csv(RACINE / "data" / "clean" / "boutiques_propres.csv")

# on garde les boutiques dont le CA est connu
fiables = b.dropna(subset=["ca_mensuel_eur"]).copy()
print("Boutiques avec un CA fiable :", len(fiables))

# en difficulte = les 3 CA les plus bas
classement = fiables.sort_values("ca_mensuel_eur")
difficulte = classement.head(3)
print(difficulte[["boutique_id", "ca_mensuel_eur"]])
print("4e plus bas :", classement.iloc[3]["boutique_id"], classement.iloc[3]["ca_mensuel_eur"])

# colonne "groupe" : difficulte ou autres
ids_difficulte = difficulte["boutique_id"].tolist()
fiables["groupe"] = np.where(fiables["boutique_id"].isin(ids_difficulte), "difficulté", "autres")

# moyennes des deux groupes
colonnes = ["surface_m2", "loyer_mensuel_eur", "passage_pietons_jour",
            "concurrents_300m", "effectif", "amplitude_horaire_h",
            "anciennete_mois", "part_emporter"]
print(fiables.groupby("groupe")[colonnes].mean().round(1).T)

# test de Mann-Whitney : l'ecart est-il du au hasard ?
print("\nP-values (Mann-Whitney) :")
for c in colonnes:
    g = fiables.loc[fiables["groupe"] == "difficulté", c].dropna()
    r = fiables.loc[fiables["groupe"] == "autres", c].dropna()
    p = stats.mannwhitneyu(g, r).pvalue
    print(c, round(p, 3))