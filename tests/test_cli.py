"""End-to-end tests: synthetic raw scans -> preprocess -> cache -> QC."""

import csv
import json

import numpy as np
import pytest

from adfusion import __version__
from adfusion import preprocessing as P
from adfusion.cli import main, qc_report
from adfusion.synthetic import make_synthetic_raw, synthetic_subject

sitk = pytest.importorskip("SimpleITK")


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    return make_synthetic_raw(tmp_path_factory.mktemp("raw"), n_subjects=4, shape=(32, 32, 32))


def test_synthetic_tree_and_labels(raw):
    raw_dir, labels = raw
    idx = P.build_file_index(raw_dir)
    assert sorted(idx) == [f"000_S_000{i}" for i in range(1, 5)]
    assert all(set(v) == {"MRI", "PET"} for v in idx.values())
    assert P.read_labels([labels]) == {
        "000_S_0001": "AD",
        "000_S_0002": "CN",
        "000_S_0003": "AD",
        "000_S_0004": "CN",
    }


def test_synthetic_pet_is_misaligned_and_registration_fixes_it():
    t1, pet = synthetic_subject((40, 40, 40), max_shift=3.0, rng=np.random.default_rng(5))
    fixed, moving = sitk.GetImageFromArray(t1), sitk.GetImageFromArray(pet)
    before = P.dice(t1, pet)
    after = P.dice(t1, P.to_array(P.register(moving, fixed, affine=False, iterations=100)))
    assert after > before


def test_preprocess_end_to_end(raw, tmp_path, capsys):
    raw_dir, labels = raw
    out = tmp_path / "cache"
    args = ["preprocess", "--raw", str(raw_dir), "--labels", str(labels), "--out", str(out),
            "--shape", "24", "--rigid-only"]  # fmt: skip
    assert main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_subjects"] == 4 and summary["labels"] == {"AD": 2, "CN": 2}
    assert summary["dice_mean"] > 0.6

    with open(out / "cache_manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["subject"] for r in rows] == [f"000_S_000{i}" for i in range(1, 5)]
    for r in rows:
        m, p = np.load(r["mri"]), np.load(r["pet"])
        assert m.shape == p.shape == (24, 24, 24)
        assert m.dtype == p.dtype == np.float32
        assert m.min() >= 0 and m.max() <= 1 and p.min() >= 0 and p.max() <= 1
        assert 0 < float(r["reg_dice"]) <= 1

    # resumed run: nothing is recomputed and Dice values are kept
    assert main(args) == 0
    again = json.loads(capsys.readouterr().out)
    assert again["dice_mean"] == pytest.approx(summary["dice_mean"])


def test_qc_report_and_strict_exit(tmp_path, capsys):
    man = tmp_path / "cache_manifest.csv"
    man.write_text(
        "subject,label,mri,pet,reg_dice\n"
        "000_S_0001,AD,a,b,0.81\n000_S_0002,CN,a,b,0.42\n000_S_0003,CN,a,b,\n"
    )
    rep = qc_report(man, threshold=0.5)
    assert rep["n_subjects"] == 3 and rep["n_with_dice"] == 2
    assert rep["below_threshold"] == ["000_S_0002"]
    assert rep["dice_min"] == 0.42
    assert main(["qc", str(man)]) == 0
    assert main(["qc", str(man), "--strict"]) == 1
    capsys.readouterr()


def test_synth_command(tmp_path, capsys):
    assert main(["synth", "--out", str(tmp_path / "s"), "--n", "2", "--shape", "16"]) == 0
    assert (tmp_path / "s" / "000_S_0002" / "Coreg_FDG" / "pet.nii.gz").exists()
    assert "2 subjects" in capsys.readouterr().out
    with pytest.raises(ValueError):
        make_synthetic_raw(tmp_path / "x", n_subjects=0)


def test_version_and_bad_shape(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert __version__ in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["synth", "--out", "x", "--shape", "4"])
