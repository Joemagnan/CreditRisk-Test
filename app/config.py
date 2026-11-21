"""Paramètres du système, fixés AVANT l'entraînement et les mesures.

Les bornes du contrat d'entrée, les critères d'acceptation du modèle, les
tolérances de non-régression et les objectifs de performance sont réunis ici.
Ils ont été écrits avant de voir le moindre résultat (voir l'historique Git),
pour éviter de les ajuster après coup.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "credit_synthetique.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "credit_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
REPORTS_DIR = ROOT / "reports"
REFERENCE_PATH = REPORTS_DIR / "reference_v1.0.0.json"
REFERENCE_CASES_PATH = ROOT / "tests" / "reference" / "cas_reference.json"

MODEL_VERSION = "1.0.0"
SEED = 2026

# Variables reçues par l'API (contrat) et cible du jeu de données
FEATURES = ["age", "revenu_mensuel", "montant_credit", "duree_mois", "anciennete_activite"]
TARGET = "defaut"

# --- Contrat d'entrée (hypothèses H1 à H6 du plan de test) -------------------
AGE_MIN, AGE_MAX = 18, 75  # H1 : majorité légale ; 75 ans = limite usuelle de l'assurance emprunteur
REVENU_MAX = 50_000_000  # H2 : FCFA par mois ; au-delà, on suppose une erreur de saisie
MONTANT_MAX = 50_000_000  # H3 : plafond retenu pour un crédit de microfinance (FCFA)
DUREE_MIN, DUREE_MAX = 1, 60  # H4 : durée en mois entiers, 5 ans au plus
ANCIENNETE_MAX = 60  # H5 : années
AGE_DEBUT_ACTIVITE = 15  # H6 : ancienneté <= âge - 15 ans

# Au-delà, l'entrée reste valide mais une alerte est renvoyée à l'agent
TAUX_EFFORT_ALERTE = 0.50  # mensualité estimée > 50 % du revenu
PERCENTILES_PLAGE = (1, 99)  # plage "habituelle" mesurée sur le jeu d'entraînement

# --- Choix du seuil de décision (sur le jeu de VALIDATION uniquement) ---------
# H7 : un défaut non détecté (FN) coûte environ 5 fois plus cher qu'un bon
# dossier envoyé à tort en étude approfondie (FP).
COUT_FAUX_NEGATIF = 5.0
COUT_FAUX_POSITIF = 1.0

# --- Critères d'acceptation du modèle (évalués une seule fois sur le test) ---
CRITERES_ACCEPTATION = {
    "rappel_min": 0.70,  # au moins 7 défauts sur 10 repérés
    "precision_min": 0.25,  # au plus 3 fausses alertes pour 1 vrai défaut
    "roc_auc_min": 0.75,
    "ecart_auc_train_test_max": 0.05,  # surapprentissage
}

# --- Non-régression : tolérances fixées avant toute comparaison ---------------
TOLERANCES_REGRESSION = {
    "baisse_rappel_max": 0.03,  # au-delà : rejet
    "baisse_f1_max": 0.03,  # au-delà : rejet
    "baisse_roc_auc_max": 0.02,  # au-delà : investigation
    "ecart_proba_meme_modele": 1e-9,  # même modèle rechargé : écart quasi nul
    "ecart_proba_investigation": 0.10,  # nouveau modèle : écart jugé important
    "part_cas_modifies_max": 0.05,  # part de cas de référence qui changent de classe
}

# --- Objectifs de performance (justification dans reports/PERFORMANCE.md) ----
OBJECTIFS_PERFORMANCE = {
    "inference_p95_ms": 10.0,
    "api_p95_ms": 200.0,  # sous 20 utilisateurs simultanés
    "utilisateurs_cible": 20,
    "debit_min_req_s": 20.0,
    "taux_erreur_max": 0.0,
}
