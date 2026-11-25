# Importation des bibliothèques

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
# Définition des chemins du projet

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "clean" / "boutiques_propres.csv"

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Chargement du fichier CSV

df = pd.read_csv(DATA_PATH)

print("Aperçu des données :")
print(df.head())

print("\nDimensions du fichier :", df.shape)
# Définition de la variable cible

# Garder uniquement les boutiques avec un CA connu
ca_connu = df.dropna(subset=["ca_mensuel_eur"]).copy()

# Calcul de la médiane du chiffre d'affaires
mediane = ca_connu["ca_mensuel_eur"].median()

# Création de la variable cible
ca_connu["marche"] = (
    ca_connu["ca_mensuel_eur"] >= mediane
).astype(int)

# Affichage des résultats
print("\n--- DEFINITION DE LA VARIABLE CIBLE ---")
print("Médiane du CA :", mediane, "euros")
print("Nombre de boutiques avec CA connu :", len(ca_connu))
print("Boutiques qui marchent (1) :", (ca_connu["marche"] == 1).sum())
print("Boutiques qui ne marchent pas (0) :", (ca_connu["marche"] == 0).sum())
# Sélection des variables explicatives

features = [
    "surface_m2",
    "passage_pietons_jour",
    "concurrents_300m",
]

# Suppression des boutiques avec des variables manquantes
data = ca_connu.dropna(subset=features).copy()

# Définition de X et y
X = data[features]
y = data["marche"]

# Affichage des informations
print("\n--- SELECTION DES VARIABLES ---")
print("Variables utilisées :", features)
print("Nombre de boutiques conservées :", len(data))
print("Nombre de boutiques exclues :", len(ca_connu) - len(data))
print("Valeurs manquantes :")
print(X.isna().sum())
print("Répartition des classes :")
print(y.value_counts().sort_index())
# Création du modèle de classification

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

print("\n--- MODELE DE CLASSIFICATION ---")
print("Modèle utilisé : Régression logistique")
print("Standardisation : StandardScaler")
print("Modèle créé avec succès.")
# Préparation de la validation croisée

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Génération des prédictions
y_pred = cross_val_predict(model, X, y, cv=cv)

print("\n--- VALIDATION CROISEE ---")
print("Méthode : Stratified K-Fold")
print("Nombre de groupes : 5")
print("Nombre de prédictions :", len(y_pred))
print("Validation croisée terminée avec succès.")
# Calcul de la matrice de confusion

cm = confusion_matrix(y, y_pred, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()

print("\n--- MATRICE DE CONFUSION ---")
print(cm)

print("Vrais négatifs (TN) :", tn)
print("Faux positifs (FP) :", fp)
print("Faux négatifs (FN) :", fn)
print("Vrais positifs (TP) :", tp)
# Calcul des indicateurs de performance

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)

print("\n--- INDICATEURS DE PERFORMANCE ---")
print("Accuracy :", round(accuracy, 3))
print("Precision :", round(precision, 3))
print("Recall :", round(recall, 3))
print("F1-score :", round(f1, 3))

print("\n--- RAPPORT DE CLASSIFICATION ---")
print(classification_report(y, y_pred, zero_division=0))
# Comparaison avec un modèle de référence

baseline = DummyClassifier(strategy="most_frequent")

baseline_pred = cross_val_predict(
    baseline,
    X,
    y,
    cv=cv
)

baseline_accuracy = accuracy_score(y, baseline_pred)

print("\n--- COMPARAISON AVEC LE MODELE DE REFERENCE ---")
print("Accuracy du modèle principal :", round(accuracy, 3))
print("Accuracy du modèle de référence :", round(baseline_accuracy, 3))
# Création du graphique de la matrice de confusion

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ne marche pas (0)", "Marche (1)"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("Q5-Q6 - Matrice de confusion")
plt.tight_layout()

# Enregistrement du graphique
plt.savefig(
    OUTPUT_DIR / "q5_q6_matrice_confusion.png",
    dpi=300
)

plt.close()

print("\nGraphique de la matrice de confusion enregistré.")
# Enregistrement des prédictions

predictions = data[["boutique_id", "ca_mensuel_eur", "marche"]].copy()
predictions["prediction"] = y_pred

predictions.to_csv(
    OUTPUT_DIR / "q5_q6_predictions.csv",
    index=False
)

print("Fichier des prédictions enregistré.")
# Enregistrement des indicateurs de performance

with open(OUTPUT_DIR / "q5_q6_indicateurs.txt", "w", encoding="utf-8") as f:
    f.write(f"Seuil de CA : {mediane} euros\n")
    f.write(f"Nombre de boutiques : {len(data)}\n")
    f.write(f"Accuracy : {accuracy:.3f}\n")
    f.write(f"Precision : {precision:.3f}\n")
    f.write(f"Recall : {recall:.3f}\n")
    f.write(f"F1-score : {f1:.3f}\n")
    f.write(f"Accuracy modèle de référence : {baseline_accuracy:.3f}\n")
    f.write(f"Matrice de confusion :\n{cm}\n")

print("Fichier des indicateurs enregistré.")