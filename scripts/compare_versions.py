"""Non-régression : compare des versions candidates à la référence figée v1.0.0.

Chaque candidate correspond à une évolution qu'une équipe pourrait proposer.
Elles sont entraînées sur les mêmes données et évaluées sur le même jeu de
validation figé et les mêmes 206 cas de référence. Le jeu de test n'est pas utilisé.

    python scripts/compare_versions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config  # noqa: E402
from app.evaluation import comparer_a_reference, metriques  # noqa: E402
from app.preprocessing import VARIABLES_MODELE, preparer_variables  # noqa: E402
from app.training import entrainer_version  # noqa: E402

CANDIDATES = [
    {
        "version": "1.0.1",
        "changement": "réentraînement à l'identique (contrôle : la chaîne est-elle stable ?)",
        "algorithme": "regression_logistique",
        "params": {"C": 1.0, "class_weight": "balanced"},
    },
    {
        "version": "1.1.0",
        "changement": "régularisation plus forte (C = 0,1)",
        "algorithme": "regression_logistique",
        "params": {"C": 0.1, "class_weight": "balanced"},
    },
    {
        "version": "1.2.0",
        "changement": "arbre de décision (profondeur 5), jugé plus lisible par les agents",
        "algorithme": "arbre_decision",
        "params": {"max_depth": 5, "min_samples_leaf": 30, "class_weight": "balanced"},
    },
    {
        "version": "1.3.0",
        "changement": "prétraitement simplifié : suppression de la variable taux_effort",
        "algorithme": "regression_logistique",
        "params": {"C": 1.0, "class_weight": "balanced"},
        "variables": [v for v in VARIABLES_MODELE if v != "taux_effort"],
    },
    {
        "version": "1.4.0",
        "changement": "retour au seuil par défaut 0,5 au lieu du seuil choisi par le coût",
        "algorithme": "regression_logistique",
        "params": {"C": 1.0, "class_weight": "balanced"},
        "seuil_force": 0.5,
    },
]


def main() -> None:
    reference = json.loads(config.REFERENCE_PATH.read_text(encoding="utf-8"))
    cas = json.loads(config.REFERENCE_CASES_PATH.read_text(encoding="utf-8"))
    entrees = preparer_variables(pd.DataFrame([c["entree"] for c in cas]))
    df = pd.read_csv(config.DATA_PATH)

    resultats = []
    for candidate in CANDIDATES:
        variables = candidate.get("variables", VARIABLES_MODELE)
        run = entrainer_version(
            df,
            candidate["version"],
            candidate["algorithme"],
            variables=variables,
            **candidate["params"],
        )
        modele = run["modele"]
        seuil = candidate.get("seuil_force", run["metadata"]["seuil"])
        validation = run["jeux"]["validation"]
        p_val = modele.predict_proba(preparer_variables(validation)[variables])[:, 1]
        p_cas = modele.predict_proba(entrees[variables])[:, 1]
        sorties = {
            "metriques": metriques(validation[config.TARGET], p_val, seuil),
            "probas_reference": p_cas.tolist(),
            "contrat_ok": bool(np.isfinite(p_cas).all() and ((p_cas >= 0) & (p_cas <= 1)).all()),
        }
        comparaison = comparer_a_reference(reference, sorties)
        resultats.append(
            {
                "version": candidate["version"],
                "changement": candidate["changement"],
                "seuil": seuil,
                "metriques_validation": sorties["metriques"],
                **comparaison,
            }
        )
        m = sorties["metriques"]
        print(
            f"v{candidate['version']:6s} seuil={seuil:.2f} rappel={m['rappel']:.3f} "
            f"F1={m['f1']:.3f} AUC={m['roc_auc']:.3f} -> {comparaison['decision'].upper()} "
            f"{'; '.join(comparaison['motifs'])}"
        )

    rapport = {
        "reference": {
            "version": reference["version"],
            "jeu": reference["jeu"],
            "metriques": reference["metriques"],
        },
        "tolerances": config.TOLERANCES_REGRESSION,
        "candidates": resultats,
    }
    (config.REPORTS_DIR / "non_regression.json").write_text(
        json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
