# Registre des anomalies

Gravité : **critique** (décision fausse ou faille, bloque le pilote), **majeure** (risque réel,
à corriger avant pilote ou à accepter explicitement), **mineure** (gêne limitée),
**outillage** (erreur dans les outils de test eux-mêmes).

| ID | Anomalie | Trouvée par | Gravité | Statut |
|---|---|---|---|---|
| AN-01 | Le premier jeu synthétique rendait le critère d'AUC inatteignable | contrôle de l'AUC maximale du générateur | majeure | corrigée |
| AN-02 | Le risque remonte légèrement avec le revenu au-delà de 2,4 M FCFA | `test_gagner_plus_n_augmente_pas_le_risque` | mineure | ouverte (test en échec attendu) |
| AN-03 | La « probabilité de défaut » renvoyée n'est pas calibrée | tableau des scénarios d'API, puis mesure sur la validation | majeure | ouverte, correction évaluée (v1.5.0) |
| AN-04 | Rappel de 0,49 sur le test, critère 0,70 non atteint | évaluation finale (`scripts/train.py`) | critique | ouverte, arbitrage métier nécessaire |
| AN-05 | 62 % de défauts non détectés après 60 ans, contre 52 % entre 25 et 60 ans | `scripts/audit_segments.py` | majeure | ouverte |
| AN-06 | Avec le gestionnaire d'erreurs par défaut de FastAPI, `Infinity` provoque une erreur 500 et les valeurs saisies sont renvoyées au client | vérification de conception | majeure | évitée (gestionnaire dédié, testé) |
| AN-07 | CSV écrit en fins de ligne Windows : empreinte SHA-256 différente après passage par Git | avertissement Git au premier commit | mineure | corrigée |
| AN-08 | Mémoire et CPU du serveur mal mesurés (5 Mo, 0 %) | incohérence des premières mesures | outillage | corrigée |
| AN-09 | Le prétraitement d'une seule demande avec pandas coûte environ 1 ms, soit 2 fois l'inférence | benchmark | mineure | ouverte (amélioration) |

## Détail

### AN-01 – Signal trop faible dans le premier jeu synthétique (corrigée)

- **Constat** : avec le premier réglage du générateur, la régression logistique obtenait une AUC
  de 0,707 sur le test et un rappel de 0,375. En calculant l'AUC de la *vraie* probabilité
  utilisée pour tirer les défauts, j'ai obtenu 0,68 : aucun modèle ne pouvait atteindre le
  critère AUC >= 0,75. Le problème venait des données, pas du modèle.
- **Correction** : effets du générateur renforcés et réglés sans aucun modèle, en visant une AUC
  maximale de 0,80 et un taux de défaut d'environ 10 %. Les critères d'acceptation n'ont pas
  changé. Le jeu est entièrement régénéré : le nouveau jeu de test n'a jamais été vu.
- **Preuve** : `reports/historique/essai1_signal_faible_evaluation.json`, commit `c624ae3`.

### AN-02 – Sens de variation du revenu (ouverte, mineure)

- **Risque visé** : un score contestable (« je gagne plus et mon risque augmente »).
- **Attendu** : à demande égale, un revenu plus élevé ne doit pas augmenter le risque.
- **Observé** : la probabilité baisse jusqu'à 2,4 M FCFA puis remonte de 0,17693 à 0,17703 à 3 M.
  Le coefficient de `log_revenu` est positif : quand le taux d'effort devient négligeable, c'est
  lui qui l'emporte.
- **Correction recommandée** : retirer `log_revenu` (l'information est déjà dans le taux
  d'effort) ou utiliser un modèle sous contrainte de monotonie. Le test est marqué
  `xfail(strict=True)` : il redeviendra bloquant dès que l'anomalie sera corrigée.

### AN-03 – Probabilités non calibrées (ouverte, majeure)

- **Observé** : la requête de l'énoncé reçoit `probabilite_defaut = 0.5465` avec
  `niveau_risque = faible`. Sur la validation, la probabilité moyenne prédite est de 39 %
  pour 9,8 % de défauts réels ; les dossiers notés entre 0,4 et 0,6 font en réalité défaut
  dans 11 % des cas. La pondération `class_weight="balanced"` multiplie les odds par 9,2.
- **Risque** : un agent qui lit « 55 % » croit le dossier très risqué alors qu'il est classé
  faible. Le champ ne veut pas dire ce que son nom annonce.
- **Correction évaluée** : diviser les odds par 9,2 (version candidate 1.5.0). Mêmes décisions
  pour les 206 cas de référence, même AUC, probabilité moyenne ramenée à 9,3 %. Le seuil
  choisi sur les probabilités corrigées vaut 0,16, ce qui correspond à la théorie (1/(1+5)).
  La règle de non-régression demande une investigation (écarts de probabilité jusqu'à 0,50),
  ce qui est voulu : ce changement doit être validé par une personne, pas accepté en silence.

### AN-04 – Rappel insuffisant (ouverte, critique)

- **Observé** : rappel 0,491 sur le test (58 défauts repérés sur 118), critère 0,70.
- **Analyse** : le seuil 0,64 est le seuil optimal pour l'hypothèse H7 (un faux négatif coûte
  5 fois un faux positif). Atteindre 70 % de rappel supposerait d'accepter beaucoup plus de
  fausses alertes : la version candidate au seuil 0,5 monte à 0,63 de rappel en validation mais
  sa précision tombe à 0,20 et son coût moyen augmente. **H7 et le critère de rappel sont
  incompatibles entre eux** pour ce modèle.
- **Décision demandée** : au métier de dire lequel des deux prime (ratio de coût réel, capacité
  d'étude approfondie des agents). Je n'ai pas changé le seuil après avoir vu le test.

### AN-05 – Profils mal détectés (ouverte, majeure)

- **Observé** (100 000 dossiers indépendants) : les plus de 60 ans font défaut deux fois plus
  souvent que les 25-60 ans (15,6 % contre 8,3 %) mais reçoivent moins d'alertes que la moyenne
  (12,8 % contre 15,1 %). Taux de faux négatifs : 62,3 % [58,6 ; 65,9] contre 52,3 %.
- **Cause** : le modèle est linéaire en âge et a appris que le risque diminue avec l'âge, ce
  qui est faux après 60 ans dans ces données.
- **Correction recommandée** : variable de tranche d'âge ou modèle non linéaire, puis suivi du
  taux de faux négatifs par tranche d'âge. Sur de vraies données, ce suivi devrait être encadré
  juridiquement (l'âge peut être un critère discriminatoire).

### AN-06 – Gestionnaire d'erreurs par défaut de FastAPI (évitée)

- **Vérification** : avec le gestionnaire par défaut, une requête contenant `Infinity` produit
  une erreur 500 (la valeur reçue ne peut pas être écrite en JSON), et une valeur invalide est
  recopiée dans la réponse (`"secret-123"` renvoyé au client).
- **Mesure** : gestionnaire dédié qui renvoie 400 ou 422 avec des messages en français, sans
  jamais reprendre la valeur reçue. Tests `test_valeur_non_finie_rejetee` (12 cas) et
  `test_messages_traduits_sans_reprendre_la_valeur`.

### AN-07 – Fins de ligne du CSV (corrigée)

pandas écrit des fins de ligne CRLF sous Windows ; Git les convertit en LF. L'empreinte des
données enregistrée dans la référence aurait différé sur le runner Linux et fait échouer
`test_les_donnees_n_ont_pas_change` en CI. Correction : `lineterminator="\n"` (commit `6585a33`).

### AN-08 – Mesure du serveur (outillage, corrigée)

Le `python.exe` d'un environnement virtuel Windows est un lanceur qui démarre l'interpréteur
dans un processus enfant. Le benchmark mesurait le lanceur : 5,1 Mo et 0 % de CPU. Après
correction (processus et enfants) : 177 Mo au repos, 186 Mo au maximum, environ 1 cœur
sous charge.

### AN-09 – Coût du prétraitement unitaire (ouverte, mineure)

Pour une demande, le pipeline complet prend 1,5 ms dont 0,47 ms pour le modèle : la création
de DataFrames pandas domine. Sans conséquence aux volumes visés (P95 API 2,8 ms en séquentiel),
mais un prétraitement NumPy diviserait ce temps si le débit devenait un enjeu.
