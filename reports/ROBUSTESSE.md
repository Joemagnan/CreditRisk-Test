# Rapport de robustesse

Suite : `tests/test_robustness.py` (exécution du 21/11/2026, modèle v1.0.0).
Règle appliquée : une entrée **invalide** est rejetée sans appeler le modèle ; une entrée
**inhabituelle mais valide** est traitée et accompagnée d'une alerte ; l'API ne répond
jamais 500 à cause d'une entrée.

| Test | Risque visé | Comportement attendu | Résultat observé | Gravité si échec | Anomalie / correction |
|---|---|---|---|---|---|
| Valeurs manquantes (`null` sur chacun des 5 champs, objet vide) | score calculé sur une donnée absente | rejet 422, champ désigné | 6/6 conformes | majeure | – |
| Valeurs non numériques (`"abc"`, `""`, `"12"`, `"1e5"`, liste, objet, booléens) sur les 5 champs | conversion silencieuse d'un texte | rejet 422 (aucune conversion implicite) | 40/40 conformes | majeure | – |
| Valeurs non finies (`NaN`, `Infinity`, `-Infinity`, `1e400`) en JSON brut | plantage, valeur infinie dans le calcul | rejet 422, jamais 500 | 12/12 conformes | majeure | AN-06 évitée : le gestionnaire par défaut de FastAPI répondait 500 |
| Limites des intervalles (8 couples valeur acceptée / refusée) | erreur « à un près » sur les bornes | borne acceptée (200), borne franchie rejetée (422) | 8/8 conformes | majeure | – |
| Entrées valides très éloignées de l'entraînement (revenu 45 M ou 1 FCFA, montant 49 M sur 60 mois, 75 ans) | score extrapolé présenté comme fiable | 200, probabilité finie dans [0, 1], au moins une alerte | 4/4 conformes | majeure | les plages habituelles (percentiles 1 et 99) sont enregistrées avec le modèle |
| Schéma incorrect (liste, objet imbriqué, noms en majuscules, tout en texte) | requête mal comprise | rejet 422 | 4/4 conformes | majeure | – |
| 150 demandes valides générées (Hypothesis) | cas limite non prévu | 200 et probabilité dans [0, 1] | 150/150 conformes | critique | – |
| 150 objets JSON quelconques générés (Hypothesis) | erreur serveur sur une entrée imprévue | 200 ou 422, jamais 500 | 150/150 conformes | majeure | – |
| Montant croissant (0,1 à 6 M FCFA, autres valeurs fixes) | score incohérent avec le bon sens | risque non décroissant | conforme | moyenne | – |
| Revenu croissant (50 000 à 3 M FCFA) | score incohérent avec le bon sens | risque non croissant | **non conforme** au-delà de 2,4 M (+0,0001) | mineure | **AN-02** : retirer `log_revenu` ou imposer la monotonie |
| Revenu ±1 % sur les 206 cas de référence | sauts de score pour une variation insignifiante | variation de probabilité < 0,02 | conforme | moyenne | – |

## Ce que ces tests ne montrent pas

- La sensibilité est mesurée autour de dossiers réalistes ; elle n'impose pas que la classe
  reste la même : un dossier proche du seuil peut légitimement basculer.
- Les alertes reposent sur des bornes fixes (percentiles du jeu d'entraînement). Elles ne
  détectent pas une dérive progressive de l'ensemble des demandes : il faudrait suivre la
  distribution des entrées en production (PSI ou test de Kolmogorov-Smirnov par variable).
- La robustesse du **modèle** sur de vraies données n'est pas établie : seule celle du
  **système** l'est.
