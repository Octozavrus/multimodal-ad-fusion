"""End-to-end tests: synthetic cohort -> nested CV -> results -> figures."""

import csv
import json

import numpy as np
import pytest
import torch

from adfusion.cli import main
from adfusion.figures import fusion_weights, make_all
from adfusion.synthetic import make_synthetic_cohort, synthetic_pair
from adfusion.train import TrainConfig, resolve_device, run_cv, train_fold
from adfusion.variants import build_mixed_manifest, build_variant

FAST = dict(
    epochs=2,
    batch_size=4,
    folds=3,
    inner_frac=0.25,
    patience=0,
    n_boot=100,
    device="cpu",
    workers=0,
)


def test_synthetic_pair_has_class_signal():
    """AD volumes are darker inside the planted regions than CN volumes."""
    rng = np.random.default_rng(0)
    g = np.meshgrid(*[np.linspace(-1, 1, 32)] * 3, indexing="ij")
    pet_roi = ((g[0] - 0.2) ** 2 + (g[1] - 0.4) ** 2 + g[2] ** 2) < 0.15**2
    mri_roi = (g[0] ** 2 + (g[1] + 0.1) ** 2 + (np.abs(g[2]) - 0.25) ** 2) < 0.08**2

    def roi_means(label):
        pairs = [synthetic_pair(label, rng=rng) for _ in range(8)]
        return np.mean([m[mri_roi].mean() for m, _ in pairs]), np.mean(
            [p[pet_roi].mean() for _, p in pairs]
        )

    (cn_m, cn_p), (ad_m, ad_p) = roi_means(0), roi_means(1)
    assert ad_m < cn_m - 0.1
    assert ad_p < cn_p - 0.1


def test_config_roundtrip_and_validation(tmp_path):
    cfg = TrainConfig(epochs=3, models=["mri"])
    f = tmp_path / "c.json"
    f.write_text(json.dumps(cfg.__dict__))
    assert TrainConfig.from_json(f) == cfg
    with pytest.raises(ValueError, match="unknown config keys"):
        TrainConfig.from_dict({"epochz": 3})
    with pytest.raises(ValueError, match="unknown models"):
        TrainConfig(models=["nope"]).validate()


@pytest.mark.slow
def test_run_cv_writes_all_outputs(records, tmp_path):
    cfg = TrainConfig(models=["mri", "hybrid_noattn"], **FAST)
    summary = run_cv(records, cfg, tmp_path)
    for f in (
        "results_per_fold.csv",
        "results_summary.csv",
        "paired_tests.csv",
        "oof_predictions.json",
        "learning_curves.json",
        "run_config.json",
    ):
        assert (tmp_path / f).exists(), f
    assert [r["model"] for r in summary] == ["mri", "hybrid_noattn"]
    for r in summary:
        assert 0 <= r["auc_pooled"] <= 1
        assert r["auc_pooled_ci_low"] <= r["auc_pooled_ci_high"]

    oof = json.loads((tmp_path / "oof_predictions.json").read_text())
    assert sorted(oof["mri"]["subject"]) == sorted(r["subject"] for r in records)
    with open(tmp_path / "paired_tests.csv") as fh:
        tests = list(csv.DictReader(fh))
    assert len(tests) == 1 and float(tests[0]["p_holm"]) >= float(tests[0]["p_raw"])
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) == 6

    figs = make_all(tmp_path, tmp_path / "figs", dpi=60)
    assert {p.name for p in figs} >= {"Fig_ROC.png", "Fig_Confusion.png", "Fig_FusionWeights.png"}
    w = fusion_weights(tmp_path / "checkpoints", "hybrid_noattn")
    assert w.shape == (3, 3)
    np.testing.assert_allclose(w.sum(1), 1.0, rtol=1e-5)
    assert len(fusion_weights(tmp_path / "checkpoints", "hybrid")) == 0  # exact stem match


@pytest.mark.slow
def test_model_learns_planted_signal(tmp_path):
    """Sanity check equivalent to the overfit check used during development."""
    torch.manual_seed(0)
    man = make_synthetic_cohort(tmp_path, n_subjects=36, shape=(32, 32, 32), effect=0.9, seed=3)
    from adfusion.data import load_manifest

    recs = load_manifest(man)
    cfg = TrainConfig(models=["pet"], epochs=12, batch_size=4, patience=0, device="cpu")
    _, yt, pt, _, hist = train_fold(
        "pet", recs[:24], recs[24:30], recs[30:], resolve_device("cpu"), cfg
    )
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(yt, pt) >= 0.8


def test_roi_and_mixed_variants(manifest, tmp_path):
    roi = build_variant(manifest, tmp_path / "roi", mode="roi", target=(16, 16, 16))
    mixed = build_mixed_manifest(manifest, roi, tmp_path / "mixed")
    with open(mixed) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 24
    assert all("roi" in r["mri"] and "roi" not in r["pet"] for r in rows)
    assert np.load(rows[0]["mri"]).shape == (16, 16, 16)
    with pytest.raises(ValueError):
        build_variant(manifest, tmp_path / "x", mode="bogus")


@pytest.mark.slow
def test_cli_end_to_end(tmp_path, capsys):
    assert main(["synth-cohort", "--out", str(tmp_path / "c"), "--n", "18", "--shape", "16"]) == 0
    args = [
        "train",
        "--manifest",
        str(tmp_path / "c" / "cache_manifest.csv"),
        "--out",
        str(tmp_path / "r"),
        "--models",
        "pet",
        "--epochs",
        "1",
        "--folds",
        "3",
        "--inner-frac",
        "0.25",
        "--batch-size",
        "4",
        "--n-boot",
        "50",
        "--device",
        "cpu",
        "--no-checkpoints",
    ]
    assert main(args) == 0
    assert "pooled AUC" in capsys.readouterr().out
    assert main(["figures", str(tmp_path / "r"), "--dpi", "50"]) == 0
    assert (tmp_path / "r" / "Fig_ROC.png").exists()


def test_cli_info_and_version(capsys):
    assert main(["info"]) == 0
    assert "hybrid_noattn" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_cli_rejects_small_shape():
    with pytest.raises(SystemExit):
        main(["synth-cohort", "--out", "x", "--shape", "8"])
