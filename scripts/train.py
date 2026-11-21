"""Entraîne la version 1.0.0, l'évalue UNE fois sur le test et fige la référence.

    python scripts/train.py            # entraîne et sauvegarde le modèle
    python scripts/train.py --figer    # (re)fige aussi la référence de non-régression

La référence n'est pas recalculée automatiquement : sans --figer, elle reste
telle qu'elle a été enregistrée, sinon elle masquerait les régressions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config  # noqa: E402
from app.evaluation import metriques, verifier_criteres  # noqa: E402
from app.prediction import sha256_fichier  # noqa: E402
from app.preprocessing import preparer_variables  # noqa: E402
from app.training import entrainer_version, sauvegarder  # noqa: E402

# Cas limites ajoutés aux cas de référence tirés de la validation
CAS_LIMITES = [
    {
        "age": 18,
        "revenu_mensuel": 60000,
        "montant_credit": 150000,
        "duree_mois": 6,
        "anciennete_activite": 0,
    },
    {
        "age": 75,
        "revenu_mensuel": 400000,
        "montant_credit": 2000000,
        "duree_mois": 12,
        "anciennete_activite": 40,
    },
    {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4,
    },
    {
        "age": 40,
        "revenu_mensuel": 150000,
        "montant_credit": 50000,
        "duree_mois": 1,
        "anciennete_activite": 10,
    },
    {
        "age": 30,
        "revenu_mensuel": 100000,
        "montant_credit": 6000000,
        "duree_mois": 60,
        "anciennete_activite": 0.5,
    },
    {
        "age": 50,
        "revenu_mensuel": 3000000,
        "montant_credit": 20000000,
        "duree_mois": 36,
        "anciennete_activite": 20,
    },
]


def taux_avec_intervalle(succes: int, n: int) -> dict:
    """Proportion et intervalle de Wilson à 95 % (utile pour les petits segments)."""
    if n == 0:
        return {"valeur": None, "ic95": None, "n": 0}
    z = 1.96
    p = succes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    demi = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return {
        "valeur": round(p, 4),
        "ic95": [round(centre - demi, 4), round(centre + demi, 4)],
        "n": n,
    }


def analyse_segments(test: pd.DataFrame, probas: np.ndarray, seuil: float) -> dict:
    """Taux de faux négatifs par profil de demandeur (jeu de test)."""
    df = test.assign(proba=probas, alerte=(probas >= seuil).astype(int))
    segments = {
        "age < 25 ans": df["age"] < 25,
        "age 25-60 ans": (df["age"] >= 25) & (df["age"] <= 60),
        "age > 60 ans": df["age"] > 60,
        "activité < 1 an": df["anciennete_activite"] < 1,
        "activité >= 1 an": df["anciennete_activite"] >= 1,
        "taux d'effort > 40 %": df["montant_credit"] / df["duree_mois"] / df["revenu_mensuel"]
        > 0.40,
        "ensemble": pd.Series(True, index=df.index),
    }
    resultat = {}
    for nom, masque in segments.items():
        seg = df[masque]
        defauts = seg[seg[config.TARGET] == 1]
        sains = seg[seg[config.TARGET] == 0]
        resultat[nom] = {
            "dossiers": int(len(seg)),
            "defauts": int(len(defauts)),
            "taux_faux_negatifs": taux_avec_intervalle(
                int((defauts["alerte"] == 0).sum()), len(defauts)
            ),
            "taux_faux_positifs": taux_avec_intervalle(
                int((sains["alerte"] == 1).sum()), len(sains)
            ),
        }
    return resultat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figer", action="store_true", help="figer la référence de non-régression")
    args = parser.parse_args()

    df = pd.read_csv(config.DATA_PATH)
    run = entrainer_version(
        df, config.MODEL_VERSION, "regression_logistique", C=1.0, class_weight="balanced"
    )
    modele, metadata = run["modele"], run["metadata"]
    seuil = metadata["seuil"]

    # Évaluation finale : le jeu de test n'est utilisé qu'ici, une seule fois
    p_test = modele.predict_proba(run["x_test"])[:, 1]
    m_test = metriques(run["y_test"], p_test, seuil)
    metadata["metriques_test"] = m_test
    acceptation = verifier_criteres(m_test, metadata["metriques_entrainement"]["roc_auc"])
    metadata["acceptation"] = acceptation

    metadata = sauvegarder(modele, metadata, config.MODEL_PATH, config.METADATA_PATH)

    evaluation = {
        "version": metadata["version"],
        "seuil": seuil,
        "metriques_entrainement": metadata["metriques_entrainement"],
        "metriques_validation": metadata["metriques_validation"],
        "metriques_test": m_test,
        "acceptation": acceptation,
        "segments_test": analyse_segments(run["jeux"]["test"], p_test, seuil),
        "coefficients": dict(
            zip(
                metadata["variables_modele"],
                (round(float(c), 4) for c in modele.named_steps["classifieur"].coef_[0]),
                strict=True,
            )
        ),
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "evaluation_v1.0.0.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.figer or not config.REFERENCE_PATH.exists():
        validation = run["jeux"]["validation"]
        echantillon = validation.sample(200, random_state=config.SEED)
        cas = [
            {k: (int(v) if k in ("age", "duree_mois") else float(v)) for k, v in ligne.items()}
            for ligne in echantillon[config.FEATURES].to_dict("records")
        ] + CAS_LIMITES
        probas_cas = modele.predict_proba(preparer_variables(pd.DataFrame(cas)))[:, 1]
        config.REFERENCE_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.REFERENCE_CASES_PATH.write_text(
            json.dumps(
                [
                    {"entree": c, "proba_attendue": float(p)}
                    for c, p in zip(cas, probas_cas, strict=True)
                ],
                indent=1,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        reference = {
            "version": metadata["version"],
            "sha256_modele": metadata["sha256"],
            "sha256_donnees": sha256_fichier(config.DATA_PATH),
            "date": metadata["date_entrainement"],
            "jeu": "validation figée (graine 2026, découpage 60/20/20 stratifié)",
            "metriques": metadata["metriques_validation"],
            "probas_reference": [float(p) for p in probas_cas],
            "contrat_ok": True,
        }
        config.REFERENCE_PATH.write_text(
            json.dumps(reference, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        print(f"référence figée : {config.REFERENCE_PATH.name} ({len(cas)} cas)")
    else:
        print(f"référence existante conservée : {config.REFERENCE_PATH.name}")

    print(f"seuil retenu (validation) : {seuil}")
    for nom in ("metriques_entrainement", "metriques_validation", "metriques_test"):
        m = evaluation[nom]
        print(
            f"{nom:24s} rappel={m['rappel']:.3f} précision={m['precision']:.3f} "
            f"F1={m['f1']:.3f} AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} {m['matrice_confusion']}"
        )
    print("critères :", acceptation)


if __name__ == "__main__":
    main()
