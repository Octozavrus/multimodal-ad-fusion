"""Cohort assembly from ADNI image-search CSV exports.

Given one export for T1 MRI and one for FDG-PET, find the subjects that have
both modalities and pick, for each subject and modality, the most processed
image (for example "Coreg, Avg, Std Img and Vox Siz, Uniform Resolution" for
PET and "GradWarp; B1; N3; Scaled" for MRI).

The exports themselves are covered by the ADNI data use agreement and must
not be committed to the repository.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ADNI_ID = re.compile(r"\d{3}_S_\d{4}")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Find a column by normalised exact name, then by substring."""
    norm = {_norm(c): c for c in df.columns}
    for c in candidates:
        if _norm(c) in norm:
            return norm[_norm(c)]
    for c in candidates:
        for k, orig in norm.items():
            if _norm(c) in k:
                return orig
    return None


def load_export(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read an export and add a canonical ``_subj`` column (NNN_S_NNNN)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    meta = {
        "subj": find_col(df, "Subject ID", "Subject", "PTID"),
        "desc": find_col(df, "Description", "Image Description", "Series Description"),
        "grp": find_col(df, "Group", "Research Group", "DX"),
    }

    def extract(row) -> str:
        if meta["subj"]:
            m = ADNI_ID.search(str(row[meta["subj"]]))
            if m:
                return m.group(0)
        for v in row.values:
            m = ADNI_ID.search(str(v))
            if m:
                return m.group(0)
        return ""

    df["_subj"] = df.apply(extract, axis=1).str.upper() if len(df) else pd.Series(dtype=str)
    return df, meta


def pet_score(desc: str) -> int:
    d = str(desc).lower()
    s = 0
    if "coreg" in d and "averaged" in d:
        s += 1
    if "standardized image and voxel size" in d or "std img and vox siz" in d:
        s += 2
    if "uniform resolution" in d:
        s += 3
    return s


def mri_score(desc: str) -> int:
    d = str(desc).lower()
    s = 0
    s += "gradwarp" in d
    s += "b1" in d
    s += 2 * ("n3" in d)
    s += "scaled" in d
    s -= "mask" in d
    return int(s)


def intersect(mri_csv: str | Path, pet_csv: str | Path) -> dict:
    """Return the usable cohort and one best image per subject and modality."""
    mri, mm = load_export(mri_csv)
    pet, pm = load_export(pet_csv)
    both = sorted((set(mri["_subj"]) & set(pet["_subj"])) - {""})

    groups = {}
    for df, meta in ((mri, mm), (pet, pm)):
        if meta["grp"]:
            sub = df[df["_subj"].isin(both)].drop_duplicates("_subj")
            groups = sub.set_index("_subj")[meta["grp"]].str.upper().to_dict()
            break

    best = []
    for df, meta, scorer, mod in ((mri, mm, mri_score, "MRI"), (pet, pm, pet_score, "PET")):
        d = df[df["_subj"].isin(both)].copy()
        if meta["desc"]:
            d["_score"] = d[meta["desc"]].map(scorer)
            d = d.sort_values(["_subj", "_score"], ascending=[True, False])
        d = d.drop_duplicates("_subj", keep="first")
        for _, r in d.iterrows():
            best.append(
                {
                    "subject": r["_subj"],
                    "modality": mod,
                    "description": r[meta["desc"]] if meta["desc"] else "",
                }
            )
    return {
        "subjects": both,
        "groups": groups,
        "best_images": sorted(best, key=lambda x: (x["subject"], x["modality"])),
        "n_mri_only": len(set(mri["_subj"]) - set(pet["_subj"]) - {""}),
        "n_pet_only": len(set(pet["_subj"]) - set(mri["_subj"]) - {""}),
    }
