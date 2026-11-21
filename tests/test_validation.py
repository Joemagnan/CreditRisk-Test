"""Tests unitaires du module de validation (contrat d'entrée)."""

import math

import pytest
from pydantic import ValidationError

from app import config
from app.validation import DemandeCredit, detecter_alertes, traduire_erreurs


def avec(demande, **changements):
    return {**demande, **changements}


def test_demande_valide_acceptee(demande_valide):
    d = DemandeCredit(**demande_valide)
    assert d.age == 35
    assert d.duree_mois == 12
    # un entier JSON est accepté pour un champ décimal
    assert d.revenu_mensuel == 250000.0
    assert isinstance(d.revenu_mensuel, float)


# --- EX01 âge, EX03 durée, EX09 montant, EX02 revenu : bornes -----------------


@pytest.mark.parametrize(
    "champ, valeur",
    [
        ("age", config.AGE_MIN),
        ("age", config.AGE_MAX),
        ("duree_mois", config.DUREE_MIN),
        ("duree_mois", config.DUREE_MAX),
        ("revenu_mensuel", 1),
        ("revenu_mensuel", config.REVENU_MAX),
        ("montant_credit", 1),
        ("montant_credit", config.MONTANT_MAX),
        ("anciennete_activite", 0),
    ],
)
def test_bornes_acceptees(demande_valide, champ, valeur):
    d = DemandeCredit(**{**demande_valide, "anciennete_activite": 0, champ: valeur})
    assert getattr(d, champ) == valeur


@pytest.mark.parametrize(
    "champ, valeur",
    [
        ("age", config.AGE_MIN - 1),
        ("age", config.AGE_MAX + 1),
        ("age", -5),
        ("duree_mois", 0),
        ("duree_mois", config.DUREE_MAX + 1),
        ("duree_mois", -12),
        ("revenu_mensuel", 0),
        ("revenu_mensuel", -1000),
        ("revenu_mensuel", config.REVENU_MAX + 1),
        ("montant_credit", 0),
        ("montant_credit", -1),
        ("montant_credit", config.MONTANT_MAX + 1),
        ("anciennete_activite", -0.1),
        ("anciennete_activite", config.ANCIENNETE_MAX + 1),
    ],
)
def test_valeurs_hors_bornes_rejetees(demande_valide, champ, valeur):
    with pytest.raises(ValidationError) as info:
        DemandeCredit(**avec(demande_valide, **{champ: valeur}))
    assert info.value.errors()[0]["loc"] == (champ,)


# --- Types ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "champ, valeur",
    [
        ("age", "35"),
        ("age", 35.5),
        ("age", 35.0),  # choix documenté : l'âge et la durée sont des entiers stricts
        ("age", True),
        ("duree_mois", 12.5),
        ("duree_mois", "12"),
        ("revenu_mensuel", "250000"),
        ("revenu_mensuel", "deux cent mille"),
        ("revenu_mensuel", True),
        ("revenu_mensuel", None),
        ("revenu_mensuel", [250000]),
        ("montant_credit", {"valeur": 1000000}),
        ("anciennete_activite", "4 ans"),
    ],
)
def test_types_incorrects_rejetes(demande_valide, champ, valeur):
    with pytest.raises(ValidationError):
        DemandeCredit(**avec(demande_valide, **{champ: valeur}))


@pytest.mark.parametrize("champ", ["revenu_mensuel", "montant_credit", "anciennete_activite"])
@pytest.mark.parametrize("valeur", [math.nan, math.inf, -math.inf])
def test_valeurs_non_finies_rejetees(demande_valide, champ, valeur):
    with pytest.raises(ValidationError):
        DemandeCredit(**avec(demande_valide, **{champ: valeur}))


# --- EX06 champs manquants, EX08 champs supplémentaires -----------------------


@pytest.mark.parametrize("champ", config.FEATURES)
def test_champ_manquant_rejete(demande_valide, champ):
    demande_valide.pop(champ)
    with pytest.raises(ValidationError) as info:
        DemandeCredit(**demande_valide)
    assert info.value.errors()[0]["type"] == "missing"


def test_champ_supplementaire_rejete(demande_valide):
    with pytest.raises(ValidationError) as info:
        DemandeCredit(**avec(demande_valide, sexe="F"))
    assert info.value.errors()[0]["type"] == "extra_forbidden"


# --- EX12 cohérence âge / ancienneté -------------------------------------------


def test_anciennete_compatible_avec_age(demande_valide):
    limite = 35 - config.AGE_DEBUT_ACTIVITE
    assert DemandeCredit(**avec(demande_valide, anciennete_activite=limite))
    with pytest.raises(ValidationError, match="incohérente avec l'âge"):
        DemandeCredit(**avec(demande_valide, anciennete_activite=limite + 0.5))


# --- Messages d'erreur -------------------------------------------------------------


def test_messages_traduits_sans_reprendre_la_valeur(demande_valide):
    with pytest.raises(ValidationError) as info:
        DemandeCredit(**avec(demande_valide, age=150, revenu_mensuel="secret-123"))
    details = traduire_erreurs(info.value.errors())
    assert {d["champ"] for d in details} == {"age", "revenu_mensuel"}
    messages = " ".join(d["message"] for d in details)
    assert "inférieur ou égal à 75" in messages
    assert "un nombre est attendu" in messages
    assert "secret-123" not in messages


def test_traduction_chemin_corps():
    assert traduire_erreurs([{"type": "missing", "loc": ("body",), "msg": "Field required"}]) == [
        {"champ": "corps", "message": "champ obligatoire absent"}
    ]


# --- Entrées valides mais inhabituelles : alerte, pas rejet ----------------------

PLAGES = {
    "age": [20.0, 60.0],
    "revenu_mensuel": [50_000.0, 1_000_000.0],
    "montant_credit": [100_000.0, 10_000_000.0],
    "duree_mois": [3.0, 48.0],
    "anciennete_activite": [0.0, 20.0],
}


def test_pas_d_alerte_pour_un_dossier_courant(demande_valide):
    assert detecter_alertes(DemandeCredit(**demande_valide), PLAGES) == []


def test_alerte_hors_plage_entrainement(demande_valide):
    alertes = detecter_alertes(DemandeCredit(**avec(demande_valide, revenu_mensuel=40_000_000)), PLAGES)
    assert [a["code"] for a in alertes] == ["hors_plage_entrainement"]
    assert alertes[0]["champ"] == "revenu_mensuel"


def test_alerte_taux_effort(demande_valide):
    # 3 000 000 / 12 mois = 250 000 par mois = 100 % du revenu
    alertes = detecter_alertes(DemandeCredit(**avec(demande_valide, montant_credit=3_000_000)), PLAGES)
    assert "taux_effort_eleve" in [a["code"] for a in alertes]
    assert "100%" in alertes[-1]["message"]
