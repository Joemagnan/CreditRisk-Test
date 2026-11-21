# Plan de test et matrice de traçabilité

## 1. Objet et périmètre

Le système estime la probabilité de défaut d'un demandeur de crédit. Quatre composants sont testés :

| Composant | Fichier | Ce qui est vérifié |
|---|---|---|
| Validation | `app/validation.py` | format, bornes et cohérence des données ; alertes sur les entrées atypiques |
| Prétraitement | `app/preprocessing.py` | construction des 6 variables du modèle, refus des données inexploitables |
| Modèle | `app/prediction.py`, `models/` | chargement contrôlé (SHA-256), contrat de sortie, reproductibilité |
| API | `app/main.py` | `GET /health`, `POST /predict`, `GET /model-info`, codes et messages d'erreur |

Hors périmètre : la décision d'octroi elle-même (l'API fournit une aide à la décision, l'agent
reste décisionnaire), l'équité réelle du modèle et la conformité réglementaire, que des données
synthétiques ne permettent pas d'évaluer.

## 2. Hypothèses métier

Elles sont fixées dans `app/config.py` **avant** l'entraînement (premier commit du dépôt).

| ID | Hypothèse |
|---|---|
| H1 | Âge entre 18 ans (majorité) et 75 ans (limite usuelle de l'assurance emprunteur) |
| H2 | Revenu mensuel strictement positif et au plus 50 000 000 FCFA (au-delà : erreur de saisie) |
| H3 | Montant strictement positif et au plus 50 000 000 FCFA (plafond retenu pour la microfinance) |
| H4 | Durée en mois entiers, de 1 à 60 mois |
| H5 | Ancienneté de l'activité entre 0 et 60 ans |
| H6 | Ancienneté au plus égale à l'âge moins 15 ans |
| H7 | Un défaut non détecté coûte environ 5 fois plus qu'un bon dossier envoyé à tort en étude approfondie |

## 3. Stratégie et priorités

Les tests sont priorisés selon la gravité des conséquences d'une défaillance :

- **P1** : la défaillance peut conduire à une décision de crédit fondée sur une valeur fausse,
  à un plantage du service ou à une fuite d'information. Ces tests bloquent la fusion d'une
  pull request.
- **P2** : la défaillance dégrade la qualité du service ou l'interprétation du score.
- **P3** : maintenabilité du code.

Niveaux de test : unitaires (validation, prétraitement), fonctionnels (API), intégration
(chaîne complète et pannes), robustesse (entrées limites, tests générés avec Hypothesis),
non-régression (référence figée), performance (benchmark local et Locust).

Une entrée **invalide** est rejetée (422, ou 400 si le JSON est mal formé) sans appeler le
modèle. Une entrée **valide mais inhabituelle** est traitée (200) avec une alerte dans la réponse.

## 4. Matrice de traçabilité

EX01 à EX07 sont les exigences du sujet ; EX08 à EX21 ont été ajoutées.
Résultats : exécution locale du 21/11/2026 (196 tests réussis, 1 échec attendu), confirmée en CI.

| ID | Exigence | Risque maîtrisé (gravité) | Prio. | Cas de test | Préconditions / données d'entrée | Résultat attendu | Résultat obtenu |
|---|---|---|---|---|---|---|---|
| EX01 | L'âge est un entier compris entre 18 et 75 ans | scorer un mineur ou une erreur de saisie (majeure) | P1 | `test_validation::test_bornes_acceptees`, `test_valeurs_hors_bornes_rejetees`, `test_types_incorrects_rejetes` ; `test_api::test_entree_invalide_rejetee[âge négatif]` ; `test_robustness::test_limites_des_intervalles` | modèle chargé ; âge = 17, 18, 75, 76, -5, 999, 35.5, "35", true | 18 et 75 acceptés ; les autres rejetés en 422, champ `age` désigné | conforme |
| EX02 | Le revenu est numérique, positif et plafonné | texte converti en silence, division par zéro dans le taux d'effort (majeure) | P1 | `test_api::test_entree_invalide_rejetee[revenu en texte]`, `[revenu nombre écrit en texte]`, `[revenu extrêmement élevé]` ; `test_robustness::test_valeur_non_numerique_rejetee` | revenu = "deux cent mille", "250000", 0, -1000, 10^12 | 422 avec un message en français, modèle non appelé | conforme |
| EX03 | La durée est un entier entre 1 et 60 mois | mensualité fausse (majeure) | P1 | `test_api::test_entree_invalide_rejetee[durée non entière]` ; `test_validation::test_valeurs_hors_bornes_rejetees` | durée = 12.5, 12.0, "12", 0, 61 | 422 « un nombre entier est attendu » ou borne | conforme |
| EX04 | La prédiction est une probabilité finie dans [0, 1] | décision prise sur une valeur absurde (critique) | P1 | `test_api::test_requete_valide` ; `test_robustness::test_toute_demande_valide_donne_une_probabilite` (150 demandes générées) ; `test_integration::test_erreur_interne_du_modele` | demandes valides quelconques ; faux modèle renvoyant 1.7 ou NaN | 200 et 0 <= p <= 1 ; sortie invalide du modèle bloquée (500) | conforme |
| EX05 | Une indisponibilité du modèle est signalée | réponses par défaut silencieuses (critique) | P1 | `test_api::test_modele_indisponible` ; `test_integration::test_api_demarre_sans_modele`, `test_fichier_modele_absent` | modèle retiré, fichier absent | `/health` 503, `/predict` 503 `modele_indisponible`, `/model-info` 503 | conforme |
| EX06 | Une entrée incomplète est rejetée avec un message approprié | score calculé sur des données manquantes (majeure) | P1 | `test_api::test_champ_obligatoire_absent` ; `test_validation::test_champ_manquant_rejete` ; `test_robustness::test_valeur_nulle_rejetee`, `test_objet_vide` | chaque champ retiré ou à `null` ; `{}` | 422 listant le ou les champs absents | conforme |
| EX07 | Une nouvelle version ne dégrade pas les critères critiques | régression silencieuse (critique) | P1 | `test_regression` (10 tests) ; `scripts/compare_versions.py` | référence v1.0.0 figée ; 206 cas de référence | rejet si rappel ou F1 baisse de plus de 0,03 ; investigation si AUC -0,02 | conforme (voir rapport de non-régression) |
| EX08 | Le montant est strictement positif et plafonné | crédit nul ou négatif scoré (majeure) | P1 | `test_api::test_entree_invalide_rejetee[montant nul]`, `[montant négatif]`, `[montant extrêmement élevé]` | montant = 0, -500 000, 10^15 | 422 sur `montant_credit` | conforme |
| EX09 | Un champ non prévu est refusé | attribut sensible glissé dans la requête, faute de frappe ignorée (majeure) | P2 | `test_api::test_champ_supplementaire_non_autorise` ; `test_robustness::test_schema_incorrect` | champ `sexe` ; noms en majuscules | 422 « champ non autorisé » | conforme |
| EX10 | L'ancienneté est cohérente avec l'âge | données incohérentes non détectées (moyenne) | P2 | `test_validation::test_anciennete_compatible_avec_age` | 35 ans, ancienneté 20 puis 20,5 ans | 20 accepté, 20,5 rejeté | conforme |
| EX11 | NaN, Infinity et JSON mal formé ne provoquent jamais d'erreur serveur | plantage, déni de service (majeure) | P1 | `test_robustness::test_valeur_non_finie_rejetee`, `test_aucune_entree_ne_provoque_d_erreur_serveur` (150 cas générés) ; `test_api::test_json_mal_forme` | `NaN`, `Infinity`, `1e400` ; JSON tronqué | 422 (valeurs) ou 400 (JSON), jamais 500 | conforme |
| EX12 | Une erreur interne ne montre aucune trace technique au client | fuite d'informations (majeure) | P1 | `test_integration::test_erreur_de_pretraitement_dans_l_api`, `test_dimensions_incompatibles_dans_l_api`, `test_erreur_interne_du_modele` | prétraitement en panne ; colonnes manquantes ; exception du modèle | 500 générique avec `id_requete` ; trace complète dans les journaux | conforme |
| EX13 | Seul un modèle intègre est chargé | modèle altéré ou code malveillant via pickle (critique) | P1 | `test_integration::test_fichier_modele_corrompu`, `test_objet_qui_n_est_pas_un_classifieur`, `test_variables_du_modele_differentes_des_metadonnees` | un octet du fichier modifié ; objet quelconque | refus de chargement, service non prêt | conforme |
| EX14 | Une entrée valide mais atypique est traitée avec une alerte | score extrapolé présenté comme fiable (majeure) | P2 | `test_robustness::test_entree_atypique_traitee_avec_alerte` ; `test_api::test_valeurs_tres_elevees_mais_autorisees` | revenu 40 M ou 1 FCFA ; 75 ans | 200 + alerte `hors_plage_entrainement` ou `taux_effort_eleve` | conforme |
| EX15 | Version et seuil identiques partout | décision non traçable (moyenne) | P2 | `test_integration::test_scenario_principal_de_bout_en_bout` ; `test_api::test_model_info` | réponse, `/model-info`, `metadata.json`, empreinte du fichier | mêmes valeurs | conforme |
| EX16 | Résultats reproductibles | décisions différentes pour un même dossier (majeure) | P1 | `test_api::test_meme_requete_meme_reponse` ; `test_regression::test_reentrainement_reproductible` ; `test_integration::test_sauvegarde_puis_rechargement_identiques` | même requête 3 fois ; réentraînement avec la graine 2026 | sorties identiques (écart < 10^-9) | conforme |
| EX17 | Le sens de variation est cohérent avec le métier | score contestable par un agent ou un client (moyenne) | P2 | `test_robustness::test_emprunter_plus_n_abaisse_pas_le_risque`, `test_gagner_plus_n_augmente_pas_le_risque`, `test_petite_variation_du_revenu` | montant de 0,1 à 6 M ; revenu de 50 000 à 3 M ; revenu ±1 % | risque croissant avec le montant, décroissant avec le revenu ; variation < 0,02 | **non conforme pour le revenu (AN-02)** ; conforme sinon |
| EX18 | P95 de l'API <= 200 ms sous 20 utilisateurs, débit >= 20 req/s, aucune erreur | service trop lent aux heures de pointe (moyenne) | P2 | `scripts/benchmark.py` (Locust) | serveur uvicorn local, paliers 1 à 40 utilisateurs | objectifs de `config.OBJECTIFS_PERFORMANCE` | conforme (P95 47,8 ms ; 471 req/s) |
| EX19 | Les journaux ne contiennent pas les données du demandeur | fuite de données personnelles (majeure) | P1 | `test_integration::test_journaux_sans_donnees_du_demandeur` | erreur provoquée avec revenu 987 654 | valeurs absentes des journaux | conforme |
| EX20 | Le code respecte le style et est couvert par les tests | code non testé, régressions (mineure) | P3 | CI : Ruff + `--cov-fail-under=90` | chaque push et pull request | Ruff sans erreur, couverture >= 90 % | conforme (97 %) |
| EX21 | Le modèle atteint les critères d'acceptation | modèle inutile ou dangereux en pilote (critique) | P1 | `scripts/train.py` (évaluation unique sur le test) | jeu de test (1 200 dossiers, 118 défauts) | rappel >= 0,70 ; précision >= 0,25 ; AUC >= 0,75 ; écart AUC <= 0,05 | **non conforme : rappel 0,49 (AN-04)** ; les 3 autres atteints |

## 5. Outils et preuves conservées

| Outil | Usage | Preuve |
|---|---|---|
| Pytest, pytest-cov | tests et couverture | `reports/tests/junit.xml` (artefact CI), couverture dans les journaux CI |
| Hypothesis | 300 demandes générées par exécution | tests `test_toute_demande_valide…` et `test_aucune_entree…` |
| FastAPI TestClient | tests de l'API sans serveur | `reports/RESULTATS_API.md` |
| uvicorn, httpx, Locust, psutil | performance | `reports/performance.json`, `reports/PERFORMANCE.md` |
| Ruff | qualité du code | étape « Qualité du code » de la CI |
| GitHub Actions | exécution à chaque modification | historique des exécutions du dépôt |

## 6. Critères d'acceptation et de rejet

- **Validation technique** : tous les tests P1 réussis en CI, couverture >= 90 %, Ruff sans erreur.
- **Modèle** : les quatre critères de EX21 sur le jeu de test, évalué une seule fois.
- **Nouvelle version** : règle de `app/evaluation.py::comparer_a_reference` (accepter,
  investiguer, rejeter), avec les tolérances de `config.TOLERANCES_REGRESSION`.
- Un critère P1 non atteint empêche le passage en pilote tant qu'il n'est pas corrigé ou
  explicitement accepté par le métier.
