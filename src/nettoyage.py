from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
CLEAN = RACINE / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)   # cree data/clean s il existe pas

b = pd.read_csv(RACINE / "data" / "Delorme_boutiques.csv")
v = pd.read_csv(RACINE / "data" / "Delorme_ventes_mensuelles.csv")
print("Avant nettoyage :", b.shape, v.shape)

#doublons 

b = b.drop_duplicates()
v = v.drop_duplicates()


#CA = -1
ids_ca_inconnu = b.loc[b["ca_mensuel_eur"] <= 0, "boutique_id"].tolist()
print("Boutiques sans CA fiable :", ids_ca_inconnu)
b.loc[b["ca_mensuel_eur"] <= 0, "ca_mensuel_eur"] = np.nan

masque = v["boutique_id"].isin(ids_ca_inconnu)
v.loc[masque, ["ca_mensuel_eur", "tickets"]] = np.nan

#surface = 0 m² 
b.loc[b["surface_m2"] == 0, "surface_m2"] = np.nan

#verifications
print("\n=== BOUTIQUES : valeurs manquantes ===")
print(b.isna().sum())

print("\n=== VENTES : valeurs manquantes ===")
print(v.isna().sum())

print("\nApres nettoyage :", b.shape, v.shape)

#sauvegarde
b.to_csv(CLEAN / "boutiques_propres.csv", index=False)
v.to_csv(CLEAN / "ventes_propres.csv", index=False)
print("Fichiers propres enregistres dans", CLEAN)
