"""Figures du rapport de qualification (reports/figures/).

    python scripts/figures.py

Les données viennent des fichiers de résultats déjà produits ; la figure du seuil
est calculée sur le jeu de VALIDATION (jamais sur le test).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config  # noqa: E402
from app.prediction import ServicePrediction  # noqa: E402
from app.preprocessing import preparer_variables  # noqa: E402
from app.training import decouper  # noqa: E402

BLEU, ORANGE = "#2a78d6", "#eb6834"
ENCRE, ENCRE_2, GRILLE, FOND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SORTIE = config.REPORTS_DIR / "figures"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": GRILLE,
        "axes.labelcolor": ENCRE_2,
        "axes.titlecolor": ENCRE,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": ENCRE_2,
        "ytick.color": ENCRE_2,
        "axes.grid": True,
        "grid.color": GRILLE,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": FOND,
        "axes.facecolor": FOND,
        "savefig.facecolor": FOND,
        "legend.frameon": False,
    }
)


def charge():
    perf = json.loads((config.REPORTS_DIR / "performance.json").read_text(encoding="utf-8"))
    paliers = perf["charge_progressive"]
    u = [p["utilisateurs"] for p in paliers]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.4, 1]})

    a.plot(u, [p["p50_ms"] for p in paliers], color=BLEU, lw=2, marker="o", ms=5, label="P50")
    a.plot(u, [p["p95_ms"] for p in paliers], color=ORANGE, lw=2, marker="o", ms=5, label="P95")
    a.axhline(200, color=ENCRE_2, lw=1, ls="--")
    a.text(1, 205, "objectif P95 : 200 ms", color=ENCRE_2, fontsize=8, va="bottom")
    for serie, cle in (("P50", "p50_ms"), ("P95", "p95_ms")):
        a.annotate(
            f"{serie} {paliers[-1][cle]:.0f} ms",
            (u[-1], paliers[-1][cle]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=ENCRE,
            fontsize=8,
        )
    a.set_xlim(0, 50)
    a.set_ylim(0, 230)
    a.set_xticks(u)
    a.set_xlabel("utilisateurs simultanés (sans pause)")
    a.set_ylabel("latence (ms)")
    a.set_title("Latence de POST /predict")
    a.legend(loc="upper left", bbox_to_anchor=(0.02, 0.8))

    pos = np.arange(len(u))
    debits = [p["debit_req_s"] for p in paliers]
    b.bar(pos, debits, color=BLEU, width=0.55, edgecolor=FOND, linewidth=2)
    for x, d in zip(pos, debits, strict=True):
        b.text(x, d + 8, f"{d:.0f}", ha="center", va="bottom", color=ENCRE, fontsize=8)
    b.set_xticks(pos, [str(v) for v in u])
    b.set_ylim(0, 540)
    b.set_xlabel("utilisateurs simultanés")
    b.set_ylabel("requêtes par seconde")
    b.set_title("Débit (1 processus uvicorn)")
    b.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(SORTIE / "charge_progressive.png", dpi=200)
    plt.close(fig)


def profils():
    seg = json.loads((config.REPORTS_DIR / "segments.json").read_text(encoding="utf-8"))["segments"]
    ordre = [
        "âge < 25 ans",
        "âge 25 à 60 ans",
        "âge > 60 ans",
        "activité < 1 an",
        "activité >= 1 an",
        "taux d'effort <= 40 %",
        "taux d'effort > 40 %",
    ]
    valeurs = [seg[n]["taux_faux_negatifs"]["valeur"] for n in ordre]
    bas = [v - seg[n]["taux_faux_negatifs"]["ic95"][0] for n, v in zip(ordre, valeurs, strict=True)]
    haut = [
        seg[n]["taux_faux_negatifs"]["ic95"][1] - v for n, v in zip(ordre, valeurs, strict=True)
    ]
    ensemble = seg["ensemble"]["taux_faux_negatifs"]["valeur"]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(ordre))[::-1]
    ax.barh(y, valeurs, color=BLEU, height=0.55, edgecolor=FOND, linewidth=2)
    ax.errorbar(valeurs, y, xerr=[bas, haut], fmt="none", ecolor=ENCRE, elinewidth=1, capsize=3)
    for yy, v, h in zip(y, valeurs, haut, strict=True):
        ax.text(v + h + 0.015, yy, f"{v * 100:.0f} %", va="center", color=ENCRE, fontsize=8)
    ax.axvline(ensemble, color=ENCRE_2, lw=1, ls="--")
    ax.text(
        ensemble + 0.01,
        y[-1] - 0.6,
        f"ensemble : {ensemble * 100:.0f} %",
        color=ENCRE_2,
        fontsize=8,
    )
    ax.set_yticks(y, [f"{n}  (n = {seg[n]['dossiers']:,})".replace(",", " ") for n in ordre])
    ax.set_xlim(0, 0.9)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("part des défauts réels non détectés (intervalle de confiance à 95 %)")
    fig.suptitle(
        "Taux de faux négatifs par profil (v1.0.0, 100 000 dossiers indépendants)",
        x=0.01,
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=ENCRE,
    )
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(SORTIE / "faux_negatifs_profils.png", dpi=200)
    plt.close(fig)


def seuil():
    service = ServicePrediction()
    service.charger()
    validation = decouper(pd.read_csv(config.DATA_PATH))[1]
    y = validation[config.TARGET].to_numpy()
    p = service.predire_proba(preparer_variables(validation))
    seuils = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rappel, precision = [], []
    for s in seuils:
        alerte = p >= s
        vp = int((alerte & (y == 1)).sum())
        rappel.append(vp / y.sum())
        precision.append(vp / alerte.sum() if alerte.sum() else np.nan)
    rappel, precision = np.array(rappel), np.array(precision)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(seuils, rappel, color=BLEU, lw=2, label="rappel")
    ax.plot(seuils, precision, color=ORANGE, lw=2, label="précision")
    retenu = service.metadata["seuil"]
    ax.axvline(retenu, color=ENCRE, lw=1)
    ax.text(
        retenu + 0.01,
        0.93,
        f"seuil retenu {retenu:.2f}".replace(".", ",") + "\n(coût minimal, FN = 5 × FP)",
        color=ENCRE,
        fontsize=8,
        va="top",
    )
    ax.axhline(config.CRITERES_ACCEPTATION["rappel_min"], color=ENCRE_2, lw=1, ls="--")
    ax.text(0.06, 0.715, "critère de rappel 0,70", color=ENCRE_2, fontsize=8)
    ax.axhline(config.CRITERES_ACCEPTATION["precision_min"], color=ENCRE_2, lw=1, ls=":")
    ax.text(0.80, 0.265, "critère de précision 0,25", color=ENCRE_2, fontsize=8)
    ax.annotate(
        "rappel",
        (0.33, rappel[seuils == 0.33][0]),
        xytext=(4, 6),
        textcoords="offset points",
        color=ENCRE,
        fontsize=8,
    )
    ax.annotate(
        "précision",
        (0.55, precision[seuils == 0.55][0]),
        xytext=(0, -14),
        textcoords="offset points",
        color=ENCRE,
        fontsize=8,
        ha="center",
    )
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("seuil de décision (probabilité renvoyée par le modèle)")
    ax.set_title("Rappel et précision selon le seuil – jeu de validation")
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(SORTIE / "seuil_validation.png", dpi=200)
    plt.close(fig)

    # seuil le plus haut qui donne encore 70 % de rappel sur la validation
    ok = seuils[rappel >= config.CRITERES_ACCEPTATION["rappel_min"]]
    s70 = float(ok.max())
    i = int(np.where(seuils == s70)[0][0])
    print(f"rappel >= 0,70 sur la validation : seuil {s70:.2f}, précision {precision[i]:.3f}")


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)
    charge()
    profils()
    seuil()
    print("figures écrites dans", SORTIE)


if __name__ == "__main__":
    main()
