
import pandas as pd

b = pd.read_csv("../data/Delorme_boutiques.csv")
v = pd.read_csv("../data/Delorme_ventes_mensuelles.csv")

# suppression des doublons
b = b.drop_duplicates()
v = v.drop_duplicates()

# traitement des valeurs incorrecte
b.loc[b["surface_m2"] == 0, "surface_m2"] = None

# verification des valeurs manquantes
print("=== BOUTIQUES ===")
print(b.isna().sum())

print("\n=== VENTES ===")
print(v.isna().sum())

# affichage des dimensions
print("\nDimensions boutiques :", b.shape)
print("Dimensions ventes :", v.shape)
