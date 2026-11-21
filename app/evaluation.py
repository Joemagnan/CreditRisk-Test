"""Métriques, choix du seuil et règle de non-régression."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app import config


def cout_erreurs(y_true, y_pred) -> float:
    """Coût moyen par dossier avec l'hypothèse H7 (FN = 5, FP = 1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (config.COUT_FAUX_NEGATIF * fn + config.COUT_FAUX_POSITIF * fp) / len(y_true)


def choisir_seuil(y_val, p_val) -> float:
    """Seuil qui minimise le coût des erreurs sur le jeu de VALIDATION.

    En cas d'égalité, on garde le seuil le plus bas (on privilégie le rappel).
    """
    y_val = np.asarray(y_val)
    p_val = np.asarray(p_val)
    candidats = np.round(np.arange(0.01, 1.0, 0.01), 2)
    couts = [cout_erreurs(y_val, (p_val >= s).astype(int)) for s in candidats]
    return float(candidats[int(np.argmin(couts))])


def metriques(y_true, probas, seuil: float) -> dict:
    """Métriques de classement au seuil donné, pour la classe « défaut » (1)."""
    y_true = np.asarray(y_true)
    probas = np.asarray(probas)
    y_pred = (probas >= seuil).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    return {
        "n": int(len(y_true)),
        "nb_defauts": int(y_true.sum()),
        "taux_defaut": round(float(y_true.mean()), 4),
        "seuil": seuil,
        "rappel": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probas)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probas)), 4),
        "accuracy": round(float((y_pred == y_true).mean()), 4),
        "taux_faux_positifs": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "taux_faux_negatifs": round(fn / (fn + tp), 4) if fn + tp else 0.0,
        "cout_moyen": round(cout_erreurs(y_true, y_pred), 4),
        "matrice_confusion": {"vn": tn, "fp": fp, "fn": fn, "vp": tp},
    }


def verifier_criteres(metriques_test: dict, auc_train: float) -> dict:
    """Critères d'acceptation fixés dans config.CRITERES_ACCEPTATION."""
    c = config.CRITERES_ACCEPTATION
    resultats = {
        "rappel": metriques_test["rappel"] >= c["rappel_min"],
        "precision": metriques_test["precision"] >= c["precision_min"],
        "roc_auc": metriques_test["roc_auc"] >= c["roc_auc_min"],
        "surapprentissage": auc_train - metriques_test["roc_auc"] <= c["ecart_auc_train_test_max"],
    }
    return {"criteres": resultats, "tous_atteints": all(resultats.values())}


def comparer_a_reference(reference: dict, candidat: dict, tol: dict | None = None) -> dict:
    """Compare une version candidate à la référence figée.

    `reference` et `candidat` contiennent :
      - "metriques" : métriques sur le même jeu de validation figé ;
      - "probas_reference" : probabilités sur les mêmes cas de référence ;
      - "contrat_ok" : sorties conformes au contrat (probabilités finies dans [0, 1]).

    Décision : "rejeter" si un critère critique (rappel, F1, contrat) se dégrade au-delà
    de la tolérance ; "investiguer" si l'AUC baisse ou si beaucoup de cas changent ;
    "accepter" sinon.
    """
    tol = tol or config.TOLERANCES_REGRESSION
    m_ref, m_new = reference["metriques"], candidat["metriques"]
    p_ref = np.asarray(reference["probas_reference"], dtype=float)
    p_new = np.asarray(candidat["probas_reference"], dtype=float)
    seuil_ref, seuil_new = m_ref["seuil"], m_new["seuil"]

    ecarts = np.abs(p_new - p_ref)
    changements_classe = float(np.mean((p_ref >= seuil_ref) != (p_new >= seuil_new)))
    delta = {
        "rappel": round(m_new["rappel"] - m_ref["rappel"], 4),
        "f1": round(m_new["f1"] - m_ref["f1"], 4),
        "roc_auc": round(m_new["roc_auc"] - m_ref["roc_auc"], 4),
        "precision": round(m_new["precision"] - m_ref["precision"], 4),
    }

    motifs_rejet = []
    if not candidat.get("contrat_ok", False):
        motifs_rejet.append("sorties non conformes au contrat")
    if delta["rappel"] < -tol["baisse_rappel_max"]:
        motifs_rejet.append(f"rappel en baisse de {-delta['rappel']:.3f}")
    if delta["f1"] < -tol["baisse_f1_max"]:
        motifs_rejet.append(f"F1 en baisse de {-delta['f1']:.3f}")

    motifs_investigation = []
    if delta["roc_auc"] < -tol["baisse_roc_auc_max"]:
        motifs_investigation.append(f"ROC-AUC en baisse de {-delta['roc_auc']:.3f}")
    if changements_classe > tol["part_cas_modifies_max"]:
        motifs_investigation.append(
            f"{changements_classe:.1%} des cas de référence changent de classe"
        )
    if float(ecarts.max(initial=0.0)) > tol["ecart_proba_investigation"]:
        motifs_investigation.append(f"écart de probabilité maximal {ecarts.max():.3f}")

    if motifs_rejet:
        decision = "rejeter"
    elif motifs_investigation:
        decision = "investiguer"
    else:
        decision = "accepter"

    return {
        "decision": decision,
        "motifs": motifs_rejet + motifs_investigation,
        "delta": delta,
        "ecart_proba_max": round(float(ecarts.max(initial=0.0)), 6),
        "ecart_proba_moyen": round(float(ecarts.mean()) if len(ecarts) else 0.0, 6),
        "part_changements_classe": round(changements_classe, 4),
    }
