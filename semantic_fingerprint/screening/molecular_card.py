from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, Lipinski, rdMolDescriptors


_IDENTITY_BEARING_FIELDS = {
    "name",
    "cid",
    "cas",
    "inchikey",
    "inchi_key",
    "role",
}


def _finite_int(value: int | float, descriptor_name: str) -> int:
    if not math.isfinite(value):
        raise ValueError(f"RDKit descriptor {descriptor_name} is not finite")
    return int(value)


def _finite_decimal(value: int | float, descriptor_name: str) -> str:
    if not math.isfinite(value):
        raise ValueError(f"RDKit descriptor {descriptor_name} is not finite")
    return f"{value:.3f}"


def _positive_fragment_counts(mol: Chem.Mol) -> str:
    counts: list[str] = []
    for name, function in sorted(Fragments.fns, key=lambda item: item[0]):
        count = _finite_int(function(mol), name)
        if count > 0:
            counts.append(f"{name}={count}")
    return "; ".join(counts) if counts else "none"


def build_molecular_card(smiles: str) -> dict[str, str | int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse the supplied SMILES")

    return {
        "canonical_smiles": Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=False,
        ),
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": _finite_decimal(
            Descriptors.MolWt(mol),
            "MolWt",
        ),
        "hbond_donors": _finite_int(Lipinski.NumHDonors(mol), "NumHDonors"),
        "hbond_acceptors": _finite_int(
            Lipinski.NumHAcceptors(mol),
            "NumHAcceptors",
        ),
        "tpsa": _finite_decimal(rdMolDescriptors.CalcTPSA(mol), "CalcTPSA"),
        "clogp": _finite_decimal(Descriptors.MolLogP(mol), "MolLogP"),
        "ring_count": _finite_int(
            rdMolDescriptors.CalcNumRings(mol),
            "CalcNumRings",
        ),
        "rotatable_bond_count": _finite_int(
            Lipinski.NumRotatableBonds(mol),
            "NumRotatableBonds",
        ),
        "fragment_counts": _positive_fragment_counts(mol),
    }


def render_molecular_card(card: Mapping[str, str | int]) -> str:
    return "\n".join(
        [
            f"Canonical SMILES: {card['canonical_smiles']}",
            f"Molecular formula: {card['molecular_formula']}",
            f"Molecular weight: {card['molecular_weight']}",
            f"Hydrogen-bond donors: {card['hbond_donors']}",
            f"Hydrogen-bond acceptors: {card['hbond_acceptors']}",
            f"Topological polar surface area: {card['tpsa']}",
            f"Calculated logP: {card['clogp']}",
            f"Ring count: {card['ring_count']}",
            f"Rotatable-bond count: {card['rotatable_bond_count']}",
            (
                "RDKit functional fragment counts: "
                f"{card['fragment_counts']}"
            ),
        ]
    )


def build_molecular_card_rows(
    candidates: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidate_ids: set[str] = set()

    for candidate in candidates:
        identity_fields = _IDENTITY_BEARING_FIELDS.intersection(candidate)
        if identity_fields:
            fields = ", ".join(sorted(identity_fields))
            raise ValueError(f"identity-bearing field is not allowed: {fields}")

        for required_field in ("candidate_id", "connectivity_smiles"):
            if required_field not in candidate:
                raise ValueError(f"missing required field: {required_field}")

        candidate_id = candidate["candidate_id"]
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)

        connectivity_smiles = candidate["connectivity_smiles"]
        card = build_molecular_card(connectivity_smiles)
        rows.append(
            {
                "candidate_id": candidate_id,
                "connectivity_smiles": connectivity_smiles,
                **card,
                "molecular_card": render_molecular_card(card),
            }
        )

    return rows
