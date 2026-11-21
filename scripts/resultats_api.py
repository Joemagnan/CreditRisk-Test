"""Tableau des résultats des scénarios d'API (PARTIE II), à partir d'appels réels.

python scripts/resultats_api.py   ->  reports/RESULTATS_API.md + resultats_api.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config  # noqa: E402
from app.main import app, service  # noqa: E402

VALIDE = {
    "age": 35,
    "revenu_mensuel": 250000,
    "montant_credit": 1000000,
    "duree_mois": 12,
    "anciennete_activite": 4,
}
SANS_REVENU = {k: v for k, v in VALIDE.items() if k != "revenu_mensuel"}

SCENARIOS = [
    ("S01", "Requête valide", {"json": VALIDE}, 200),
    ("S02", "Champ obligatoire absent (revenu)", {"json": SANS_REVENU}, 422),
    ("S03", "Âge négatif (-5)", {"json": {**VALIDE, "age": -5}}, 422),
    (
        "S04",
        "Revenu sous forme de texte",
        {"json": {**VALIDE, "revenu_mensuel": "deux cent mille"}},
        422,
    ),
    ("S05a", "Montant nul", {"json": {**VALIDE, "montant_credit": 0}}, 422),
    ("S05b", "Montant négatif", {"json": {**VALIDE, "montant_credit": -500000}}, 422),
    ("S06", "Durée non entière (12,5)", {"json": {**VALIDE, "duree_mois": 12.5}}, 422),
    ("S07", "Champ supplémentaire (sexe)", {"json": {**VALIDE, "sexe": "F"}}, 422),
    ("S08a", "Revenu extrêmement élevé (10^12)", {"json": {**VALIDE, "revenu_mensuel": 1e12}}, 422),
    (
        "S08b",
        "Revenu très élevé mais autorisé (40 M)",
        {"json": {**VALIDE, "revenu_mensuel": 40_000_000}},
        200,
    ),
    ("S09", "Modèle indisponible", {"json": VALIDE, "sans_modele": True}, 503),
    ("S10a", "JSON mal formé", {"content": '{"age": 35, "revenu_mensuel": '}, 400),
    ("S10b", "Corps qui n'est pas un objet", {"content": "[1, 2, 3]"}, 422),
    (
        "S10c",
        "Valeur Infinity (JSON non standard)",
        {"content": json.dumps(VALIDE).replace("250000", "Infinity")},
        422,
    ),
]


def resume_reponse(corps: dict) -> str:
    if "probabilite_defaut" in corps:
        alertes = ", ".join(a["code"] for a in corps["alertes"]) or "aucune"
        return (
            f"probabilite_defaut={corps['probabilite_defaut']:.4f}, niveau_risque={corps['niveau_risque']}, "
            f"alertes={alertes}"
        )
    texte = f"erreur={corps.get('erreur')}"
    if corps.get("details"):
        texte += " ; " + " ; ".join(f"{d['champ']} : {d['message']}" for d in corps["details"])
    return texte


def main() -> None:
    lignes = []
    with TestClient(app) as client:
        for code, scenario, requete, attendu in SCENARIOS:
            modele = service.modele
            if requete.get("sans_modele"):
                service.modele = None
            try:
                if "json" in requete:
                    r = client.post("/predict", json=requete["json"])
                else:
                    r = client.post(
                        "/predict",
                        content=requete["content"],
                        headers={"content-type": "application/json"},
                    )
            finally:
                service.modele = modele
            lignes.append(
                {
                    "id": code,
                    "scenario": scenario,
                    "code_attendu": attendu,
                    "code_obtenu": r.status_code,
                    "conforme": r.status_code == attendu,
                    "reponse": resume_reponse(r.json()),
                }
            )

    md = [
        "# Résultats des scénarios de l'API",
        "",
        "Généré par `scripts/resultats_api.py` (appels réels à l'API, modèle v1.0.0).",
        "",
        "| ID | Scénario | Code attendu | Code obtenu | Réponse | Conforme |",
        "|---|---|---|---|---|---|",
    ]
    for ligne in lignes:
        md.append(
            f"| {ligne['id']} | {ligne['scenario']} | {ligne['code_attendu']} | {ligne['code_obtenu']} "
            f"| {ligne['reponse']} | {'oui' if ligne['conforme'] else 'NON'} |"
        )
    config.REPORTS_DIR.joinpath("RESULTATS_API.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    config.REPORTS_DIR.joinpath("resultats_api.json").write_text(
        json.dumps(lignes, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for ligne in lignes:
        print(
            ligne["id"],
            ligne["code_obtenu"],
            "ok" if ligne["conforme"] else "NON CONFORME",
            ligne["reponse"],
        )


if __name__ == "__main__":
    main()
