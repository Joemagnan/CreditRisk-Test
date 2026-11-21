# Données

`credit_synthetique.csv` : 6 000 demandes de crédit **synthétiques**, générées par
`scripts/generate_data.py` (graine 2026). Le fichier est versionné car il est petit
(environ 200 Ko) et ne contient aucune donnée personnelle.

| Colonne | Type | Description |
|---|---|---|
| `age` | entier | âge du demandeur (18 à 75 ans) |
| `revenu_mensuel` | décimal | revenu mensuel en FCFA |
| `montant_credit` | décimal | montant demandé en FCFA |
| `duree_mois` | entier | durée du crédit (3 à 60 mois) |
| `anciennete_activite` | décimal | ancienneté de l'activité en années |
| `defaut` | 0 / 1 | 1 = défaut de remboursement simulé |

Taux de défaut : 9,8 % (587 défauts). AUC de la vraie probabilité de défaut : 0,800 :
c'est le plafond qu'aucun modèle ne peut dépasser sur ces données.

## Comment le défaut est simulé

Le défaut est tiré au hasard selon une probabilité qui dépend :

- du **taux d'effort** (mensualité estimée / revenu) : c'est l'effet principal ;
- de la **durée** (un peu plus de risque pour les crédits longs) ;
- de l'**ancienneté de l'activité** (moins de risque avec l'expérience, et un
  surcroît de risque pour les activités de moins d'un an) ;
- de l'**âge** (plus de risque avant 25 ans et après 60 ans).

Les effets de l'âge et de l'activité récente ne sont pas linéaires : une régression
logistique ne peut pas les reproduire exactement. C'est voulu, pour pouvoir étudier
des profils de demandeurs moins bien détectés (question de synthèse).

## Ce que ces données ne permettent pas

Elles servent à **tester le système** : validation des entrées, API, pipeline,
non-régression, performance. Comme la règle de défaut a été écrite à la main, les
métriques obtenues ne disent rien de la performance d'un vrai modèle de crédit, ni
de son équité, ni de sa conformité réglementaire.
