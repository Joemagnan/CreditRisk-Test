# Résultats des scénarios de l'API

Généré par `scripts/resultats_api.py` (appels réels à l'API, modèle v1.0.0).

| ID | Scénario | Code attendu | Code obtenu | Réponse | Conforme |
|---|---|---|---|---|---|
| S01 | Requête valide | 200 | 200 | probabilite_defaut=0.5465, niveau_risque=faible, alertes=aucune | oui |
| S02 | Champ obligatoire absent (revenu) | 422 | 422 | erreur=donnees_invalides ; revenu_mensuel : champ obligatoire absent | oui |
| S03 | Âge négatif (-5) | 422 | 422 | erreur=donnees_invalides ; age : doit être supérieur ou égal à 18 | oui |
| S04 | Revenu sous forme de texte | 422 | 422 | erreur=donnees_invalides ; revenu_mensuel : un nombre est attendu | oui |
| S05a | Montant nul | 422 | 422 | erreur=donnees_invalides ; montant_credit : doit être strictement supérieur à 0 | oui |
| S05b | Montant négatif | 422 | 422 | erreur=donnees_invalides ; montant_credit : doit être strictement supérieur à 0 | oui |
| S06 | Durée non entière (12,5) | 422 | 422 | erreur=donnees_invalides ; duree_mois : un nombre entier est attendu | oui |
| S07 | Champ supplémentaire (sexe) | 422 | 422 | erreur=donnees_invalides ; sexe : champ non autorisé | oui |
| S08a | Revenu extrêmement élevé (10^12) | 422 | 422 | erreur=donnees_invalides ; revenu_mensuel : doit être inférieur ou égal à 50 000 000 | oui |
| S08b | Revenu très élevé mais autorisé (40 M) | 200 | 200 | probabilite_defaut=0.2410, niveau_risque=faible, alertes=hors_plage_entrainement | oui |
| S09 | Modèle indisponible | 503 | 503 | erreur=modele_indisponible | oui |
| S10a | JSON mal formé | 400 | 400 | erreur=requete_mal_formee | oui |
| S10b | Corps qui n'est pas un objet | 422 | 422 | erreur=donnees_invalides ; corps : le corps de la requête doit être un objet JSON | oui |
| S10c | Valeur Infinity (JSON non standard) | 422 | 422 | erreur=donnees_invalides ; revenu_mensuel : un nombre fini est attendu (NaN et Infinity sont refusés) | oui |
