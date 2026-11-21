"""Entraînement d'une version du modèle (utilisé par scripts/train.py et la non-régression)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app import config
from app.evaluation import choisir_seuil, metriques
from app.prediction import sha256_fichier
from app.preprocessing import VARIABLES_MODELE, preparer_variables


def decouper(df: pd.DataFrame, seed: int = config.SEED):
    """60 % entraînement, 20 % validation, 20 % test, stratifiés sur la cible.

    Le découpage est fait AVANT toute transformation apprise (le StandardScaler
    est dans le pipeline et n'est ajusté que sur l'entraînement).
    """
    train, reste = train_test_split(df, test_size=0.4, stratify=df[config.TARGET], random_state=seed)
    validation, test = train_test_split(
        reste, test_size=0.5, stratify=reste[config.TARGET], random_state=seed
    )
    return train, validation, test


def construire_modele(algorithme: str = "regression_logistique", **params) -> Pipeline:
    if algorithme == "regression_logistique":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight=params.get("class_weight", "balanced"),
            C=params.get("C", 1.0),
            random_state=config.SEED,
        )
    elif algorithme == "arbre_decision":
        clf = DecisionTreeClassifier(
            max_depth=params.get("max_depth", 5),
            min_samples_leaf=params.get("min_samples_leaf", 30),
            class_weight=params.get("class_weight", "balanced"),
            random_state=config.SEED,
        )
    else:
        raise ValueError(f"algorithme inconnu : {algorithme}")
    return Pipeline([("normalisation", StandardScaler()), ("classifieur", clf)])


def plages_entrainement(train: pd.DataFrame) -> dict[str, list[float]]:
    bas, haut = config.PERCENTILES_PLAGE
    return {
        nom: [float(np.percentile(train[nom], bas)), float(np.percentile(train[nom], haut))]
        for nom in config.FEATURES
    }


def entrainer_version(df: pd.DataFrame, version: str, algorithme: str, **params) -> dict:
    """Entraîne une version complète : modèle, seuil choisi sur la validation, métriques."""
    train, validation, test = decouper(df)
    x_train, y_train = preparer_variables(train), train[config.TARGET]
    x_val, y_val = preparer_variables(validation), validation[config.TARGET]
    x_test, y_test = preparer_variables(test), test[config.TARGET]

    modele = construire_modele(algorithme, **params)
    modele.fit(x_train, y_train)

    p_train = modele.predict_proba(x_train)[:, 1]
    p_val = modele.predict_proba(x_val)[:, 1]
    seuil = choisir_seuil(y_val, p_val)

    metadata = {
        "version": version,
        "date_entrainement": date.today().isoformat(),
        "algorithme": algorithme,
        "parametres": params,
        "donnees": "SYNTHÉTIQUES (scripts/generate_data.py) : servent à tester le système, "
        "pas à démontrer la performance d'un vrai modèle de crédit",
        "graine": config.SEED,
        "variables_entree": config.FEATURES,
        "variables_modele": VARIABLES_MODELE,
        "seuil": seuil,
        "methode_seuil": "coût minimal sur la validation (FN = 5, FP = 1)",
        "tailles": {"entrainement": len(train), "validation": len(validation), "test": len(test)},
        "plages_entrainement": plages_entrainement(train),
        "metriques_entrainement": metriques(y_train, p_train, seuil),
        "metriques_validation": metriques(y_val, p_val, seuil),
    }
    return {
        "modele": modele,
        "metadata": metadata,
        "jeux": {"train": train, "validation": validation, "test": test},
        "x_test": x_test,
        "y_test": y_test,
    }


def sauvegarder(modele, metadata: dict, model_path: Path, metadata_path: Path) -> dict:
    """Écrit le modèle puis ses métadonnées, avec l'empreinte SHA-256 du fichier."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modele, model_path)
    metadata = {**metadata, "sha256": sha256_fichier(model_path)}
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata
