import pytest

from adfusion.cli import main
from adfusion.cohort import find_col, intersect, mri_score, pet_score

MRI_CSV = """Image Data ID,Subject,Group,Sex,Age,Description
I1,000_S_0001,AD,F,74,MPRAGE
I2,000_S_0001,AD,F,74,MPR; GradWarp; B1 Correction; N3; Scaled
I3,000_S_0002,CN,M,70,MPR; GradWarp; N3; Scaled
I4,000_S_0003,CN,M,71,MPRAGE
"""
PET_CSV = """Image Data ID,Subject,Group,Description
P1,000_S_0001,AD,Coreg Avg Std Img and Vox Siz Uniform Resolution
P2,000_S_0001,AD,Co-registered Dynamic
P3,000_S_0002,CN,Coreg Averaged
P4,000_S_0009,AD,Coreg Averaged
"""


@pytest.fixture
def exports(tmp_path):
    m, p = tmp_path / "mri.csv", tmp_path / "pet.csv"
    m.write_text(MRI_CSV)
    p.write_text(PET_CSV)
    return m, p


def test_scores_prefer_most_processed():
    assert mri_score("MPR; GradWarp; B1 Correction; N3; Scaled") > mri_score("MPRAGE")
    assert mri_score("N3 Scaled Mask") < mri_score("N3 Scaled")
    assert pet_score("Coreg, Avg, Std Img and Vox Siz, Uniform Resolution") > pet_score(
        "Coreg, Averaged"
    )


def test_find_col_variants():
    import pandas as pd

    df = pd.DataFrame(columns=["Subject ID", "Research Group"])
    assert find_col(df, "subject_id") == "Subject ID"
    assert find_col(df, "Group") == "Research Group"
    assert find_col(df, "Age") is None


def test_intersect(exports):
    res = intersect(*exports)
    assert res["subjects"] == ["000_S_0001", "000_S_0002"]
    assert res["groups"] == {"000_S_0001": "AD", "000_S_0002": "CN"}
    assert res["n_mri_only"] == 1 and res["n_pet_only"] == 1
    best = {(b["subject"], b["modality"]): b["description"] for b in res["best_images"]}
    assert "B1 Correction" in best[("000_S_0001", "MRI")]
    assert "Uniform Resolution" in best[("000_S_0001", "PET")]


def test_cohort_cli(exports, tmp_path, capsys):
    assert main(["cohort", str(exports[0]), str(exports[1]), "--out", str(tmp_path / "o")]) == 0
    assert "both modalities: 2" in capsys.readouterr().out
    assert (tmp_path / "o" / "best_image_per_subject.csv").exists()
