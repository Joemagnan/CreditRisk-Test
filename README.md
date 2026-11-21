# CreditRisk-Test

[![Tests CreditRisk](https://github.com/Joemagnan/CreditRisk-Test/actions/workflows/tests.yml/badge.svg)](https://github.com/Joemagnan/CreditRisk-Test/actions/workflows/tests.yml)

Assurance qualité et tests d'un système de prédiction du risque de défaut de crédit
(microfinance). Évaluation formative n° 2 de test logiciel, Joe MAGNAN, MDSIA 2.

> **Données synthétiques.** Les 6 000 demandes sont générées par `scripts/generate_data.py`
> selon une règle écrite à la main ([data/README.md](data/README.md)). Elles servent à tester le
> système, pas à mesurer la performance d'un vrai modèle de crédit.

## En bref

| | |
|---|---|
| Modèle v1.0.0 | régression logistique pondérée, seuil 0,64 choisi sur la validation (coût FN = 5 × FP) |
| Test (1 200 dossiers, 118 défauts) | 58 défauts repérés, 60 manqués, 114 fausses alertes |
| Rappel / précision / ROC-AUC | 0,491 / 0,337 / 0,799 : **critère de rappel (0,70) non atteint** |
| Tests | 196 réussis, 1 échec attendu (anomalie connue AN-02), couverture 97 % |
| Performance | P95 de l'API 47,8 ms sous 20 utilisateurs simultanés, 471 req/s, 0 erreur |
| Anomalies | 9 enregistrées : 3 corrigées, 1 évitée, 5 ouvertes ([registre](docs/ANOMALIES.md)) |
| Conclusion | système **techniquement validé** ; modèle **non apte** à un pilote en l'état |

## Structure

```
CreditRisk-Test/
├── app/
│   ├── main.py            API FastAPI : /health, /predict, /model-info
│   ├── validation.py      contrat d'entrée, alertes sur les entrées atypiques
│   ├── preprocessing.py   variables du modèle (log, taux d'effort)
│   ├── prediction.py      chargement contrôlé (SHA-256) et calcul du score
│   ├── evaluation.py      métriques, choix du seuil, règle de non-régression
│   ├── training.py        découpage 60/20/20, entraînement, sauvegarde
│   └── config.py          bornes, critères, tolérances et objectifs (fixés avant les mesures)
├── tests/
│   ├── test_validation.py   unitaires : contrat d'entrée
│   ├── test_api.py          fonctionnels : 10 scénarios de l'énoncé
│   ├── test_integration.py  bout en bout et pannes
│   ├── test_robustness.py   entrées limites, Hypothesis, sensibilité
│   ├── test_regression.py   référence figée
│   └── reference/           206 cas de référence
├── scripts/               génération, entraînement, comparaison, audit, benchmark, Locust
├── data/                  jeu synthétique (versionné) + README
├── models/                modèle v1.0.0 + métadonnées (empreinte SHA-256)
├── reports/               résultats et rapports
├── docs/                  plan de test, registre des anomalies
└── .github/workflows/tests.yml
```

## Exécution locale

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows ; source .venv/bin/activate sous Linux
pip install -r requirements.txt
```

Tests et qualité (ce que fait la CI) :

```bash
pytest --cov=app --cov-report=term-missing
ruff check . && ruff format --check .
```

API :

```bash
uvicorn app.main:app --port 8010
# http://127.0.0.1:8010/docs
curl -X POST http://127.0.0.1:8010/predict -H "Content-Type: application/json" \
     -d '{"age":35,"revenu_mensuel":250000,"montant_credit":1000000,"duree_mois":12,"anciennete_activite":4}'
```

Reproduire les résultats :

```bash
python scripts/generate_data.py        # données (graine 2026)
python scripts/train.py                # modèle v1.0.0 et évaluation finale
python scripts/compare_versions.py     # non-régression : 6 versions candidates
python scripts/audit_segments.py       # faux négatifs par profil (100 000 dossiers)
python scripts/resultats_api.py        # tableau des scénarios de l'API
pip install -r requirements-perf.txt
python scripts/benchmark.py            # performance (environ 2 min, utilise le port 8011)
```

## Contrat de l'API

`POST /predict`

```json
{"age": 35, "revenu_mensuel": 250000, "montant_credit": 1000000, "duree_mois": 12, "anciennete_activite": 4}
```

| Cas | Code | Corps |
|---|---|---|
| demande valide | 200 | `probabilite_defaut`, `niveau_risque` (`faible` / `eleve`), `seuil_decision`, `version_modele`, `alertes` |
| données invalides | 422 | `erreur: donnees_invalides`, `details: [{champ, message}]` |
| JSON mal formé | 400 | `erreur: requete_mal_formee` |
| modèle indisponible | 503 | `erreur: modele_indisponible` |
| erreur interne | 500 | message générique + `id_requete` (trace complète dans les journaux du serveur) |

Bornes : âge entier de 18 à 75 ans ; revenu et montant strictement positifs, au plus
50 000 000 FCFA ; durée entière de 1 à 60 mois ; ancienneté de 0 à 60 ans et au plus âge - 15.

## Intégration continue

`.github/workflows/tests.yml` exécute Ruff puis Pytest avec la couverture (seuil 90 %) à
chaque push et pull request vers `main`. Les tests utilisent le modèle versionné : pas de
réentraînement complet à chaque modification. Un test en échec fait échouer le workflow et
bloque la fusion (branche `main` protégée).

## Documents

- [Plan de test et matrice de traçabilité](docs/PLAN_DE_TEST.md)
- [Registre des anomalies](docs/ANOMALIES.md)
- [Résultats des scénarios de l'API](reports/RESULTATS_API.md)
- [Robustesse](reports/ROBUSTESSE.md) · [Non-régression](reports/NON_REGRESSION.md) · [Performance](reports/PERFORMANCE.md)
- Rapport de qualification : document Word remis séparément
