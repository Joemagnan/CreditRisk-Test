# Rapport de non-régression

## Référence

- Version **1.0.0**, figée le 21/11/2026 (`reports/reference_v1.0.0.json`) : empreintes SHA-256
  du modèle et des données, métriques sur le jeu de validation figé (1 200 dossiers,
  117 défauts), probabilités attendues pour 206 cas de référence (200 tirés de la validation
  et 6 cas limites).
- La référence n'est jamais recalculée par les tests. La refiger demande une commande
  explicite (`python scripts/train.py --figer`) et un commit dédié.
- Le jeu de test n'est pas utilisé pour comparer des versions.

## Tolérances (fixées avant la comparaison, `app/config.py`)

| Contrôle | Tolérance | Si dépassée |
|---|---|---|
| Sorties conformes au contrat (probabilités finies dans [0, 1]) | aucune exception | rejeter |
| Baisse du rappel | 0,03 | rejeter |
| Baisse du F1 | 0,03 | rejeter |
| Baisse de la ROC-AUC | 0,02 | investiguer |
| Cas de référence qui changent de classe | 5 % | investiguer |
| Écart de probabilité maximal (nouveau modèle) | 0,10 | investiguer |
| Écart de probabilité (même modèle rechargé) | 10^-9 | échec du test |

## Contrôles automatiques à chaque pull request (`tests/test_regression.py`)

| Test | Vérifie | Résultat |
|---|---|---|
| `test_le_modele_versionne_est_celui_de_la_reference` | le fichier du modèle n'a pas été remplacé | réussi |
| `test_les_donnees_n_ont_pas_change` | le jeu de données est identique | réussi |
| `test_predictions_identiques_sur_les_cas_de_reference` | une modification du prétraitement ou du chargement ne change aucune prédiction | réussi (écart 0) |
| `test_metriques_conformes_a_la_reference` | mêmes métriques sur la validation, décision « accepter » | réussi |
| `test_sorties_de_l_api_conformes_au_contrat` | les 206 cas passent par l'API et respectent le contrat | réussi |
| `test_reentrainement_reproductible` | réentraîner avec la graine 2026 redonne le même modèle | réussi |
| `test_sauvegarde_puis_rechargement_identiques` (intégration) | sauvegarde et rechargement sans écart | réussi |
| `test_une_version_degradee_est_rejetee` | un vrai modèle entraîné sans pondération est rejeté | réussi (rejet sur le rappel) |
| 3 tests de la règle | contrat cassé = rejet, AUC seule = investigation, baisse tolérée = accepter | réussis |

## Comparaison de six versions candidates (`scripts/compare_versions.py`)

Métriques sur le jeu de validation figé ; référence : rappel 0,410, F1 0,339, AUC 0,754,
coût moyen 0,386.

| Version | Changement proposé | Seuil | Rappel | F1 | AUC | Cas qui changent de classe | Décision | Motif |
|---|---|---|---|---|---|---|---|---|
| 1.0.1 | réentraînement à l'identique | 0,64 | 0,410 | 0,339 | 0,754 | 0 % | **accepter** | identique à la référence |
| 1.1.0 | régularisation plus forte (C = 0,1) | 0,62 | 0,444 | 0,338 | 0,755 | 0,5 % | **accepter** | dans les tolérances |
| 1.2.0 | arbre de décision, profondeur 5 | 0,70 | 0,427 | 0,339 | 0,715 | 9,2 % | **investiguer** | AUC -0,040 ; probabilités très différentes (jusqu'à 0,69) |
| 1.3.0 | suppression de la variable `taux_effort` | 0,64 | 0,419 | 0,315 | 0,735 | 6,8 % | **investiguer** | 6,8 % des cas changent de classe |
| 1.4.0 | seuil par défaut 0,5 | 0,50 | 0,632 | 0,301 | 0,754 | 16,0 % | **rejeter** | F1 -0,038 |
| 1.5.0 | probabilités recalibrées (AN-03) | 0,16 | 0,419 | 0,345 | 0,754 | 0 % | **investiguer** | écarts de probabilité jusqu'à 0,50 |

## Lecture

- **1.0.1** confirme que la chaîne est stable : même code, mêmes données, même résultat.
- **1.3.0** est instructive : supprimer la variable principale ne fait perdre que 0,024 de F1,
  car le modèle reconstitue presque le taux d'effort à partir du logarithme du montant, de
  celui du revenu et de la durée. La règle ne la rejette pas mais demande une analyse :
  6,8 % des dossiers de référence changent de décision.
- **1.4.0** améliore le rappel (+0,22) mais elle est rejetée : la précision tombe à 0,20 et le
  coût moyen passe de 0,386 à 0,430. La règle protège l'hypothèse H7 ; c'est aussi la preuve
  que le choix du seuil est une décision métier (voir AN-04).
- **1.5.0** montre l'intérêt de la décision « investiguer » : les décisions et l'AUC sont
  identiques, mais toutes les probabilités changent. Le changement est voulu ; il doit être
  accepté par une personne qui a lu ce rapport, puis la référence refigée.

**Décision** : la version 1.0.0 reste la référence. 1.1.0 pourrait la remplacer sans risque
mais n'apporte rien de décisif. 1.5.0 est recommandée après validation métier, car elle corrige
AN-03 sans changer les décisions.
