"""Faux négatifs par profil de demandeur (question de synthèse).

Le jeu de test ne contient que 118 défauts : trop peu pour comparer des profils
(7 défauts seulement chez les plus de 60 ans). Comme les données sont synthétiques,
on peut tirer un grand échantillon INDÉPENDANT avec le même générateur et une autre
graine, et y mesurer le modèle v1.0.0 figé, sans rien réentraîner.

    python scripts/audit_segments.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config  # noqa: E402
from app.prediction import ServicePrediction  # noqa: E402
from app.preprocessing import preparer_variables  # noqa: E402
from scripts.generate_data import generer  # noqa: E402
from scripts.train import taux_avec_intervalle  # noqa: E402

TAILLE_AUDIT = 100_000
GRAINE_AUDIT = 7  # différente de la graine d'entraînement (2026)


def main() -> None:
    service = ServicePrediction()
    service.charger()
    seuil = service.metadata["seuil"]

    df, _ = generer(TAILLE_AUDIT, GRAINE_AUDIT)
    p = service.predire_proba(preparer_variables(df))
    df = df.assign(alerte=(p >= seuil).astype(int))
    taux_effort = df["montant_credit"] / df["duree_mois"] / df["revenu_mensuel"]

    segments = {
        "ensemble": np.ones(len(df), dtype=bool),
        "âge < 25 ans": df["age"] < 25,
        "âge 25 à 60 ans": (df["age"] >= 25) & (df["age"] <= 60),
        "âge > 60 ans": df["age"] > 60,
        "activité < 1 an": df["anciennete_activite"] < 1,
        "activité >= 1 an": df["anciennete_activite"] >= 1,
        "taux d'effort <= 40 %": taux_effort <= 0.40,
        "taux d'effort > 40 %": taux_effort > 0.40,
    }
    resultat = {}
    for nom, masque in segments.items():
        seg = df[masque]
        defauts = seg[seg[config.TARGET] == 1]
        sains = seg[seg[config.TARGET] == 0]
        alertes = seg[seg["alerte"] == 1]
        resultat[nom] = {
            "dossiers": int(len(seg)),
            "taux_defaut_reel": round(float(seg[config.TARGET].mean()), 4),
            "taux_alerte": round(float(seg["alerte"].mean()), 4),
            "taux_faux_negatifs": taux_avec_intervalle(
                int((defauts["alerte"] == 0).sum()), len(defauts)
            ),
            "taux_faux_positifs": taux_avec_intervalle(
                int((sains["alerte"] == 1).sum()), len(sains)
            ),
            "precision": taux_avec_intervalle(
                int((alertes[config.TARGET] == 1).sum()), len(alertes)
            ),
        }
        r = resultat[nom]
        print(
            f"{nom:24s} n={r['dossiers']:6d} défaut réel={r['taux_defaut_reel']:.3f} "
            f"alertes={r['taux_alerte']:.3f} FNR={r['taux_faux_negatifs']['valeur']:.3f} "
            f"{r['taux_faux_negatifs']['ic95']} FPR={r['taux_faux_positifs']['valeur']:.3f}"
        )

    sortie = {
        "modele": service.metadata["version"],
        "seuil": seuil,
        "echantillon": f"{TAILLE_AUDIT} dossiers synthétiques indépendants (graine {GRAINE_AUDIT})",
        "segments": resultat,
    }
    (config.REPORTS_DIR / "segments.json").write_text(
        json.dumps(sortie, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
