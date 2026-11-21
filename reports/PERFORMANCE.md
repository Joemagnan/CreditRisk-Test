# Rapport de performance

Mesures du 21/11/2026 avec `scripts/benchmark.py` ; données brutes : `reports/performance.json`.

## Objectifs et justification

Les objectifs ont été fixés dans `app/config.py` avant les mesures.

Hypothèse d'usage (H8) : environ 200 agents de crédit. Aux heures de pointe, chacun évalue
au plus un dossier par minute, puisque la saisie d'une demande prend plusieurs minutes. Cela
fait au plus **3,3 requêtes par seconde**. L'agent attend la réponse devant le client : en
dessous de 200 ms, elle paraît immédiate.

| Objectif | Valeur | Raison |
|---|---|---|
| P95 de l'inférence du modèle | <= 10 ms | le modèle ne doit représenter qu'une petite part du temps de réponse |
| P95 de l'API sous 20 utilisateurs simultanés | <= 200 ms | réponse perçue comme immédiate, même dans une situation bien plus chargée que la réalité |
| Débit sous 20 utilisateurs | >= 20 req/s | 6 fois la pointe estimée |
| Taux d'erreur sous charge | 0 % | aucune demande perdue |

## Environnement de mesure

| Élément | Valeur |
|---|---|
| Machine | ordinateur portable, Intel Core Ultra 9 185H (16 cœurs physiques, 22 logiques), 64 Go de RAM |
| Système | Windows 11 (10.0.26300) ; projet sur un disque externe |
| Logiciels | Python 3.12.10 ; numpy 2.5.4 ; pandas 3.1.1 ; scikit-learn 1.9.1 ; FastAPI 0.162.2 ; pydantic 2.14.3 ; uvicorn 0.54.0 ; Locust 2.47.0 |
| Serveur | uvicorn, **1 processus**, adresse locale 127.0.0.1 (pas de réseau, pas de TLS) |
| Client de charge | Locust sur la même machine ; chaque utilisateur envoie ses demandes **sans pause** |
| Répétitions | 2 000 mesures après 100 d'échauffement (modèle, pipeline) ; 1 000 requêtes HTTP séquentielles ; 5 paliers de 15 s (la première seconde de chaque palier est écartée) |
| Méthode | `time.perf_counter` ; percentiles NumPy ; mémoire et CPU du serveur relevés toutes les 0,5 s avec psutil |

## Résultats

### Sans charge

| Mesure | Moyenne | P50 | P95 | P99 |
|---|---|---|---|---|
| Inférence du modèle (demande déjà transformée) | 0,47 ms | 0,44 ms | **0,57 ms** | 0,73 ms |
| Pipeline dans le processus (validation + prétraitement + modèle) | 1,53 ms | 1,49 ms | 1,78 ms | 2,16 ms |
| API par HTTP, requêtes séquentielles (1 000) | 2,50 ms | 2,43 ms | **2,82 ms** | 3,10 ms |

- Débit séquentiel par HTTP : 400 requêtes/s, aucune erreur.
- Inférence par lots : 5 000 demandes en 0,53 ms.
- Taille du modèle : 1,8 Ko. Mémoire du serveur : 177 Mo au repos, 186 Mo au maximum sous charge.

Une requête HTTP prend 2,5 ms, dont 0,47 ms pour le modèle. Le reste se répartit entre la
validation, le prétraitement avec pandas (AN-09) et le traitement HTTP.

### Charge progressive (Locust)

| Utilisateurs simultanés | P50 | P95 | P99 | Débit | Erreurs |
|---|---|---|---|---|---|
| 1 | 2,2 ms | 2,6 ms | 3,2 ms | 431 req/s | 0 % |
| 5 | 10,6 ms | 15,9 ms | 24,4 ms | 442 req/s | 0 % |
| 10 | 21,0 ms | 24,2 ms | 26,7 ms | 473 req/s | 0 % |
| **20** | 41,8 ms | **47,8 ms** | 53,4 ms | **471 req/s** | **0 %** |
| 40 | 82,7 ms | 99,1 ms | 119,0 ms | 473 req/s | 0 % |

## Interprétation

- **Les quatre objectifs sont atteints**, avec une marge importante.
- Le débit plafonne vers 470 requêtes/s dès 10 utilisateurs : l'unique processus uvicorn
  occupe alors un cœur entier (CPU relevé à environ 110 %). Au-delà, les requêtes attendent
  leur tour et la latence augmente proportionnellement au nombre d'utilisateurs
  (40 / 473 ≈ 85 ms, ce qui correspond au P50 mesuré à 40 utilisateurs : loi de Little).
- En extrapolant, le P95 dépasserait 200 ms vers 80 utilisateurs qui envoient des demandes
  sans pause, soit plus de 100 fois la pointe estimée. Lancer plusieurs processus
  (`--workers 4`) relèverait ce plafond.

## Limites

- Client et serveur tournent sur la même machine, sans réseau : en production, il faut ajouter
  la latence réseau, le TLS et l'authentification.
- Un ordinateur portable n'est pas un serveur de production. Les mesures ne sont comparables
  qu'entre elles, sur ce poste.
- Les tests de charge ne tournent pas en CI : un runner GitHub mutualisé donne des temps
  variables. J'ai fait la même erreur au TP 1, où un test de latence a échoué en CI pour cette raison.
