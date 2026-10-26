from __future__ import annotations

import math

import pytest

from semantic_fingerprint.screening.molecular_card import (
    build_molecular_card,
    build_molecular_card_rows,
    render_molecular_card,
)


CARD_KEYS = [
    "canonical_smiles",
    "molecular_formula",
    "molecular_weight",
    "hbond_donors",
    "hbond_acceptors",
    "tpsa",
    "clogp",
    "ring_count",
    "rotatable_bond_count",
    "fragment_counts",
]
IDENTITY_BEARING_FIELDS = {
    "name",
    "cid",
    "cas",
    "inchikey",
    "inchi_key",
    "role",
}


def test_builds_exact_deterministic_card_and_renders_only_structure() -> None:
    card = build_molecular_card("N#CC1CCOC1")

    assert card == {
        "canonical_smiles": "N#CC1CCOC1",
        "molecular_formula": "C5H7NO",
        "molecular_weight": "97.117",
        "hbond_donors": 0,
        "hbond_acceptors": 2,
        "tpsa": "33.020",
        "clogp": "0.546",
        "ring_count": 1,
        "rotatable_bond_count": 0,
        "fragment_counts": "fr_NH0=1; fr_ether=1; fr_nitrile=1",
    }
    rendered = render_molecular_card(card)
    assert rendered == "\n".join(
        [
            "Canonical SMILES: N#CC1CCOC1",
            "Molecular formula: C5H7NO",
            "Molecular weight: 97.117",
            "Hydrogen-bond donors: 0",
            "Hydrogen-bond acceptors: 2",
            "Topological polar surface area: 33.020",
            "Calculated logP: 0.546",
            "Ring count: 1",
            "Rotatable-bond count: 0",
            (
                "RDKit functional fragment counts: "
                "fr_NH0=1; fr_ether=1; fr_nitrile=1"
            ),
        ]
    )
    for forbidden_text in ("14631-44-8", "10219385", "OANDNN", "target"):
        assert forbidden_text not in rendered


def test_stereoisomers_share_nonisomeric_canonical_smiles() -> None:
    clockwise = build_molecular_card("C[C@H](O)F")
    counterclockwise = build_molecular_card("C[C@@H](O)F")

    assert clockwise["canonical_smiles"] == counterclockwise["canonical_smiles"]


def test_invalid_smiles_is_rejected() -> None:
    with pytest.raises(ValueError, match="RDKit"):
        build_molecular_card("not-a-smiles")


def test_rows_are_anonymous_deterministic_and_do_not_mutate_input() -> None:
    candidates = [
        {
            "candidate_id": "C0001",
            "connectivity_smiles": "N#CC1CCOC1",
            "source_batch": "frozen",
        }
    ]
    original = [dict(row) for row in candidates]

    rows = build_molecular_card_rows(candidates)

    assert candidates == original
    assert list(rows[0]) == [
        "candidate_id",
        "connectivity_smiles",
        *CARD_KEYS,
        "molecular_card",
    ]
    assert rows[0]["molecular_card"] == render_molecular_card(
        {key: rows[0][key] for key in CARD_KEYS}
    )
    assert IDENTITY_BEARING_FIELDS.isdisjoint(rows[0])
    assert "source_batch" not in rows[0]


@pytest.mark.parametrize("identity_field", sorted(IDENTITY_BEARING_FIELDS))
def test_rows_reject_identity_fields(identity_field: str) -> None:
    candidate = {
        "candidate_id": "C0001",
        "connectivity_smiles": "CCO",
        identity_field: "",
    }

    with pytest.raises(ValueError, match="identity-bearing field"):
        build_molecular_card_rows([candidate])


def test_rows_reject_duplicate_ids_and_accept_empty_input() -> None:
    assert build_molecular_card_rows([]) == []
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        build_molecular_card_rows(
            [
                {"candidate_id": "C0001", "connectivity_smiles": "CCO"},
                {"candidate_id": "C0001", "connectivity_smiles": "CCN"},
            ]
        )


@pytest.mark.parametrize("missing_field", ["candidate_id", "connectivity_smiles"])
def test_rows_reject_missing_fields(missing_field: str) -> None:
    candidate = {"candidate_id": "C0001", "connectivity_smiles": "CCO"}
    del candidate[missing_field]

    with pytest.raises(ValueError, match=missing_field):
        build_molecular_card_rows([candidate])


def test_multicomponent_smiles_is_not_filtered_by_card_rendering() -> None:
    card = build_molecular_card("CCO.O")

    assert card["canonical_smiles"] == "CCO.O"
    for key in (
        "molecular_weight",
        "hbond_donors",
        "hbond_acceptors",
        "tpsa",
        "clogp",
        "ring_count",
        "rotatable_bond_count",
    ):
        assert math.isfinite(float(card[key]))
