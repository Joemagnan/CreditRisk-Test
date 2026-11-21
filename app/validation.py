"""Module de validation : format et cohérence d'une demande de crédit.

Deux niveaux de contrôle, à ne pas confondre :
- une entrée INVALIDE est rejetée (l'API répond 422 sans calculer de score) ;
- une entrée VALIDE MAIS INHABITUELLE est acceptée, avec une alerte pour l'agent.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app import config


class DemandeCredit(BaseModel):
    """Contrat d'entrée de POST /predict."""

    # extra="forbid" : un champ inconnu est refusé (faute de frappe, injection de champ)
    # strict=True : pas de conversion implicite ("250000" ou true ne sont pas des nombres)
    # allow_inf_nan=False : NaN et Infinity sont refusés
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    age: int = Field(ge=config.AGE_MIN, le=config.AGE_MAX, description="Âge en années")
    revenu_mensuel: float = Field(gt=0, le=config.REVENU_MAX, description="Revenu mensuel (FCFA)")
    montant_credit: float = Field(gt=0, le=config.MONTANT_MAX, description="Montant demandé (FCFA)")
    duree_mois: int = Field(ge=config.DUREE_MIN, le=config.DUREE_MAX, description="Durée (mois)")
    anciennete_activite: float = Field(
        ge=0, le=config.ANCIENNETE_MAX, description="Ancienneté de l'activité (années)"
    )

    @model_validator(mode="after")
    def anciennete_coherente_avec_age(self) -> DemandeCredit:
        limite = self.age - config.AGE_DEBUT_ACTIVITE
        if self.anciennete_activite > limite:
            raise ValueError(
                f"anciennete_activite incohérente avec l'âge : au plus {limite} ans "
                f"pour un demandeur de {self.age} ans"
            )
        return self


def detecter_alertes(demande: DemandeCredit, plages: dict[str, list[float]]) -> list[dict]:
    """Repère les valeurs valides mais éloignées de ce que le modèle a vu à l'entraînement.

    `plages` donne, pour chaque variable, les percentiles 1 et 99 du jeu d'entraînement
    (enregistrés dans les métadonnées du modèle).
    """
    alertes = []
    valeurs = demande.model_dump()
    for champ, (bas, haut) in plages.items():
        valeur = valeurs[champ]
        if valeur < bas or valeur > haut:
            alertes.append(
                {
                    "code": "hors_plage_entrainement",
                    "champ": champ,
                    "message": (
                        f"{champ} = {valeur:g} est en dehors de la plage habituelle "
                        f"des données d'entraînement ({bas:g} à {haut:g}) : "
                        "score moins fiable, vérification recommandée"
                    ),
                }
            )

    taux_effort = demande.montant_credit / demande.duree_mois / demande.revenu_mensuel
    if taux_effort > config.TAUX_EFFORT_ALERTE:
        alertes.append(
            {
                "code": "taux_effort_eleve",
                "champ": "montant_credit",
                "message": (
                    f"la mensualité estimée représente {taux_effort:.0%} du revenu mensuel "
                    f"(seuil d'alerte : {config.TAUX_EFFORT_ALERTE:.0%})"
                ),
            }
        )
    return alertes


# Messages d'erreur en français pour les agents (Pydantic les produit en anglais)
_MESSAGES = {
    "missing": "champ obligatoire absent",
    "extra_forbidden": "champ non autorisé",
    "int_type": "un nombre entier est attendu",
    "float_type": "un nombre est attendu",
    "finite_number": "un nombre fini est attendu (NaN et Infinity sont refusés)",
    "greater_than": "doit être strictement supérieur à {gt}",
    "greater_than_equal": "doit être supérieur ou égal à {ge}",
    "less_than_equal": "doit être inférieur ou égal à {le}",
    "model_type": "le corps de la requête doit être un objet JSON",
    "model_attributes_type": "le corps de la requête doit être un objet JSON",
}


def traduire_erreurs(erreurs: list[dict]) -> list[dict]:
    """Transforme les erreurs Pydantic en liste {champ, message}.

    La valeur reçue n'est volontairement PAS renvoyée : elle peut contenir des
    données personnelles, ou des valeurs (NaN, Infinity) que JSON ne sait pas écrire.
    """
    details = []
    for erreur in erreurs:
        chemin = [str(p) for p in erreur.get("loc", ()) if p != "body"]
        champ = ".".join(chemin) or "corps"
        modele = _MESSAGES.get(erreur.get("type", ""))
        if modele:
            message = modele.format(**erreur.get("ctx", {}))
        else:
            message = str(erreur.get("msg", "valeur invalide")).removeprefix("Value error, ")
        details.append({"champ": champ, "message": message})
    return details
