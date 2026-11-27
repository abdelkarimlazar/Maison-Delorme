# Maison Delorme — Projet Datamining (Sujet 3)

Analyse du réseau de 62 boulangeries parisiennes **Maison Delorme** en vue de valider huit ouvertures pour l'an prochain.

## Structure du projet

```text
Maison-Delorme/
├── data/
│   ├── Delorme_boutiques.csv           # Données brutes des boutiques (64 lignes)
│   ├── Delorme_ventes_mensuelles.csv   # Ventes mensuelles brutes (768 lignes)
│   └── clean/
│       ├── boutiques_propres.csv       # Boutiques nettoyées (62 lignes, 14 colonnes)
│       └── ventes_propres.csv          # Ventes nettoyées (744 lignes, 4 colonnes)
├── src/
│   ├── nettoyage.py                    # Nettoyage et dédoublonnage des données
│   ├── q1_difficulte.py                # Q1 : Profil des 3 boutiques en difficulté (Mann-Whitney)
│   ├── q2_loyer.py                     # Q2 : Corrélation loyer / passage / CA
│   ├── q3_q4_regression.py             # Q3-Q4 : Régression linéaire et erreur moyenne (CV 5-fold)
│   └── q5_q6_classification.py         # Q5-Q6 : Classification logistique et matrice de confusion
├── outputs/
│   ├── q3_q4_indicateurs.txt
│   ├── q5_q6_indicateurs.txt
│   ├── q5_q6_predictions.csv
│   ├── q5_q6_matrice_confusion.png
│   └── figures/
│       ├── q2_loyer.png
│       └── q3_q4_regression.png
└── Sujet_3_Maison_Delorme (1).pdf      # Énoncé du projet
```

## Prérequis

Python 3 avec les bibliothèques suivantes :

```bash
pip install pandas numpy matplotlib scikit-learn scipy
```

## Exécution

Lancer d'abord le script de nettoyage, puis les scripts d'analyse depuis la racine du projet :

```bash
python3 src/nettoyage.py
python3 src/q1_difficulte.py
python3 src/q2_loyer.py
python3 src/q3_q4_regression.py
python3 src/q5_q6_classification.py
```

## État d'avancement

- [x] **Nettoyage (`src/nettoyage.py`)** : dédoublonnage par `boutique_id` et `(boutique_id, mois)`, remplacement des `ca_mensuel_eur <= 0`, `tickets <= 0` et `surface_m2 <= 0` par `NaN`.
- [x] **Q1 (`src/q1_difficulte.py`)** : comparaison des 3 boutiques au CA le plus bas vs le reste du réseau (test de Mann-Whitney).
- [x] **Q2 (`src/q2_loyer.py`)** : corrélations de Pearson/Spearman entre loyer, surface, passage piéton et CA.
- [x] **Q3–Q4 (`src/q3_q4_regression.py`)** : régression linéaire (CV 5-fold : `R² = 0.793`, `MAE = 2 214 €`, `MAPE = 13.8 %`) et prédiction du local proposé (`18 216 €`).
- [x] **Q5–Q6 (`src/q5_q6_classification.py`)** : régression logistique (CV stratifiée 5-fold) et matrice de confusion *(à compléter avec la prédiction du local proposé et le réglage du seuil selon le risque FP/FN)*.
- [ ] **Q7–Q8** : clustering des boutiques (nombre de modèles et format optimal).
- [ ] **Q9** : analyse géographique (carte, concentration, concurrence interne, zones vides).
- [ ] **Synthèse finale** : rapport sans code destiné à la direction commerciale.
