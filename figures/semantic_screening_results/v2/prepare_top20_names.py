#!/usr/bin/env python3
"""Create post-ranking chemical-name annotations for the final Top 20.

The output is presentation metadata only. Candidate scores and ranks are read
from the immutable publication-facing tables and are never recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from rdkit import Chem

PROPERTY_FIELDS = (
    "Title,IUPACName,MolecularFormula,ConnectivitySMILES,InChIKey"
)
PUBCHEM_PROPERTY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{namespace}/{identifier}"
    f"/property/{PROPERTY_FIELDS}/JSON"
)
PUBCHEM_SIMILARITY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/"
    "smiles/{smiles}/cids/JSON?Threshold=100"
)
RETRIEVAL_DATE = "2026-07-24"


def canonical_connectivity(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(
    session: requests.Session,
    url: str,
    retries: int = 5,
    *,
    optional: bool = False,
) -> dict | None:
    for attempt in range(retries):
        response = session.get(url, timeout=30)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 404:
            return None
        if response.status_code == 429 or response.status_code >= 500:
            wait = min(2**attempt, 16)
            time.sleep(wait)
            continue
        raise RuntimeError(f"PubChem request failed ({response.status_code}): {url}")
    if optional:
        return None
    raise RuntimeError(f"PubChem request exhausted retries: {url}")


def properties_from_payload(payload: dict | None) -> list[dict]:
    if payload is None:
        return []
    return list(payload.get("PropertyTable", {}).get("Properties", []))


def matching_property(records: list[dict], expected_smiles: str) -> dict | None:
    expected = canonical_connectivity(expected_smiles)
    matches: list[dict] = []
    for record in records:
        returned = record.get("ConnectivitySMILES")
        if returned and canonical_connectivity(str(returned)) == expected:
            matches.append(record)
    if not matches:
        return None
    return sorted(matches, key=lambda item: int(item.get("CID") or 0))[0]


def fetch_annotation(
    session: requests.Session,
    smiles: str,
) -> tuple[dict, str]:
    encoded_smiles = quote(smiles, safe="")
    direct_url = PUBCHEM_PROPERTY_URL.format(
        namespace="smiles",
        identifier=encoded_smiles,
    )
    direct = matching_property(
        properties_from_payload(get_json(session, direct_url)),
        smiles,
    )
    if direct and (direct.get("Title") or direct.get("IUPACName")):
        return direct, "pubchem_smiles"

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse Top-20 SMILES: {smiles}")
    inchi_key = Chem.MolToInchiKey(molecule)
    key_url = PUBCHEM_PROPERTY_URL.format(
        namespace="inchikey",
        identifier=quote(inchi_key, safe=""),
    )
    key_match = matching_property(
        properties_from_payload(get_json(session, key_url)),
        smiles,
    )
    if key_match and (key_match.get("Title") or key_match.get("IUPACName")):
        return key_match, "pubchem_inchikey"
    if key_match:
        direct = key_match

    similarity_url = PUBCHEM_SIMILARITY_URL.format(smiles=encoded_smiles)
    similarity_payload = get_json(session, similarity_url, optional=True)
    candidate_cids = (
        similarity_payload.get("IdentifierList", {}).get("CID", [])
        if similarity_payload
        else []
    )
    for cid in sorted(int(value) for value in candidate_cids):
        cid_url = PUBCHEM_PROPERTY_URL.format(namespace="cid", identifier=cid)
        record = matching_property(
            properties_from_payload(get_json(session, cid_url)),
            smiles,
        )
        if record and (record.get("Title") or record.get("IUPACName")):
            return record, "pubchem_connectivity_match"

    if direct:
        return direct, "pubchem_unnamed_structure"
    return {
        "CID": 0,
        "ConnectivitySMILES": smiles,
        "InChIKey": inchi_key,
    }, "canonical_smiles_only"


def display_name(
    record: dict,
    source: str,
    *,
    selected: bool,
    fallback_smiles: str,
) -> tuple[str, str]:
    if selected:
        return "3-cyanotetrahydrofuran", "post_ranking_selected_name"

    title = str(record.get("Title") or "").strip()
    iupac = str(record.get("IUPACName") or "").strip()
    name = title or iupac
    if name:
        if source == "pubchem_connectivity_match":
            stripped = re.sub(r"^\([^)]+\)-", "", name)
            if stripped != name:
                return stripped, "pubchem_name_stereo_descriptor_removed"
        return name, "pubchem_title" if title else "pubchem_iupac_name"

    return fallback_smiles, "canonical_smiles_fallback"


def build_annotations(data_dir: Path) -> pd.DataFrame:
    top20 = pd.read_csv(data_dir / "anonymous" / "top20_axis_matrix.csv")
    ranking = pd.read_csv(data_dir / "annotated" / "ranking_all_annotated.csv")
    selected_ids = set(
        ranking.loc[ranking["is_selected_candidate"].astype(bool), "candidate_id"]
    )
    if len(top20) != 20 or top20["final_rank"].tolist() != list(range(1, 21)):
        raise ValueError("Top-20 source must contain final ranks 1–20.")
    if len(selected_ids) != 1:
        raise ValueError("Exactly one post-ranking selected candidate is required.")

    rows: list[dict[str, object]] = []
    with requests.Session() as session:
        session.headers.update(
            {"User-Agent": "organic-additive-semantic-fingerprint/figure-annotation"}
        )
        for item in top20.itertuples(index=False):
            record, query_source = fetch_annotation(
                session,
                str(item.connectivity_smiles),
            )
            returned_smiles = str(
                record.get("ConnectivitySMILES") or item.connectivity_smiles
            )
            if canonical_connectivity(returned_smiles) != canonical_connectivity(
                str(item.connectivity_smiles)
            ):
                raise ValueError(
                    f"PubChem connectivity mismatch for {item.candidate_id}"
                )
            label, label_source = display_name(
                record,
                query_source,
                selected=str(item.candidate_id) in selected_ids,
                fallback_smiles=str(item.connectivity_smiles),
            )
            rows.append(
                {
                    "final_rank": int(item.final_rank),
                    "candidate_id": str(item.candidate_id),
                    "connectivity_smiles": str(item.connectivity_smiles),
                    "pubchem_cid": int(record.get("CID") or 0),
                    "pubchem_title": str(record.get("Title") or ""),
                    "pubchem_iupac_name": str(record.get("IUPACName") or ""),
                    "molecular_formula": str(record.get("MolecularFormula") or ""),
                    "inchi_key": str(record.get("InChIKey") or ""),
                    "display_name": label,
                    "display_name_source": label_source,
                    "query_source": query_source,
                    "identity_annotation_stage": "after_deterministic_final_ranking",
                    "identity_used_for_ranking": False,
                }
            )
            time.sleep(0.12)

    annotations = pd.DataFrame(rows).sort_values("final_rank").reset_index(drop=True)
    if annotations["candidate_id"].tolist() != top20["candidate_id"].tolist():
        raise ValueError("Annotation order disagrees with the final Top 20.")
    if annotations["display_name"].eq("").any():
        raise ValueError("Every Top-20 candidate must have a concrete display label.")
    return annotations


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Prepare post-ranking Top-20 chemical-name annotations."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir.parent / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "data",
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "top20_identity_annotations.csv"
    metadata_path = output_dir / "top20_identity_metadata.json"

    annotations = build_annotations(args.data_dir.resolve())
    csv_text = annotations.to_csv(index=False, lineterminator="\n")
    if csv_path.exists() and not args.replace:
        if csv_path.read_text(encoding="utf-8") != csv_text:
            raise FileExistsError(
                f"Existing annotation differs; use --replace after review: {csv_path}"
            )
    else:
        csv_path.write_text(csv_text, encoding="utf-8")

    metadata = {
        "schema_version": 1,
        "retrieval_date": RETRIEVAL_DATE,
        "source": "PubChem PUG REST",
        "annotation_stage": "after_deterministic_final_ranking",
        "identity_used_for_ranking": False,
        "row_count": len(annotations),
        "annotation_csv_sha256": sha256(csv_path),
    }
    metadata_text = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if metadata_path.exists() and not args.replace:
        if metadata_path.read_text(encoding="utf-8") != metadata_text:
            raise FileExistsError(
                f"Existing metadata differs; use --replace after review: {metadata_path}"
            )
    else:
        metadata_path.write_text(metadata_text, encoding="utf-8")

    print(csv_path)
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
