from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
b = pd.read_csv(RACINE / "data" / "clean" / "boutiques_propres.csv")

#le CA connu
fiables = b.dropna(subset=["ca_mensuel_eur"]).copy()
print("Boutiques avec un CA fiable :", len(fiables))    

# 3 CA le plus bas 
classement = fiables.sort_values("ca_mensuel_eur")
difficulte = classement.head(3)
print(difficulte[["boutique_id", "ca_mensuel_eur"]])
print("4e plus bas :", classement.iloc[3]["boutique_id"], classement.iloc[3]["ca_mensuel_eur"])

#creer une colonne "groupe"
ids_difficulte = difficulte["boutique_id"].tolist()
fiables["groupe"] = np.where(fiables["boutique_id"].isin(ids_difficulte), "difficulté", "autres")

#comparer les moyennes des deux groupes 
colonnes = ["surface_m2", "loyer_mensuel_eur", "passage_pietons_jour",
            "concurrents_300m", "effectif", "amplitude_horaire_h",
            "anciennete_mois", "part_emporter"]
print(fiables.groupby("groupe")[colonnes].mean().round(1).T)
