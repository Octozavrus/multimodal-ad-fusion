import numpy as np
import pytest
import torch

from adfusion.data import (
    PairedVolumeDataset,
    Split,
    assert_disjoint,
    load_manifest,
    make_loaders,
    nested_splits,
)


def test_manifest_keeps_only_ad_and_cn(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("subject,label,mri,pet\nA,AD,a_m,a_p\nB,MCI,b_m,b_p\nC,cn,c_m,c_p\n")
    recs = load_manifest(f)
    assert [r["subject"] for r in recs] == ["A", "C"]
    assert [r["y"] for r in recs] == [1, 0]


def test_manifest_data_root(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("subject,label,mri,pet\nA,AD,/old/A_MRI.npy,/old/A_PET.npy\n")
    r = load_manifest(f, data_root="/new")[0]
    assert r["mri"].replace("\\", "/") == "/new/A_MRI.npy"


def test_manifest_missing_column(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("subject,label,mri\nA,AD,x\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_manifest(f)


def test_manifest_duplicate_subjects(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("subject,label,mri,pet\nA,AD,x,y\nA,CN,x,y\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(f)


def test_dataset_item_shapes(records):
    mri, pet, y = PairedVolumeDataset(records)[0]
    assert mri.shape == pet.shape == (1, 16, 16, 16)
    assert mri.dtype == torch.float32 and y.dtype == torch.float32


def test_augmentation_flips_both_modalities_together(records):
    ds_plain = PairedVolumeDataset(records)
    ds_flip = PairedVolumeDataset(records, augment=True, flip_p=1.0, noise_std=0.0)
    m0, p0, _ = ds_plain[1]
    m1, p1, _ = ds_flip[1]
    torch.testing.assert_close(m1, torch.flip(m0, dims=[1]))
    torch.testing.assert_close(p1, torch.flip(p0, dims=[1]))


def test_augmentation_noise_stays_in_range(records):
    m, p, _ = PairedVolumeDataset(records, augment=True, noise_std=0.2)[0]
    assert m.min() >= 0 and m.max() <= 1 and p.min() >= 0 and p.max() <= 1


def test_loaders_batch(records):
    tr, ev = make_loaders(records[:8], records[8:12], batch_size=4)
    mri, pet, y = next(iter(tr))
    assert mri.shape == (4, 1, 16, 16, 16) and y.shape == (4,)
    assert len(ev.dataset) == 4


@pytest.mark.parametrize("n,folds,repeats", [(24, 3, 1), (50, 5, 2), (168, 5, 1)])
def test_nested_splits_are_disjoint_and_cover_all(n, folds, repeats):
    y = np.array([i % 2 for i in range(n)])
    splits = list(nested_splits(y, folds, 0.2, seed=42, repeats=repeats))
    assert len(splits) == folds * repeats
    for rep in range(repeats):
        tests = np.concatenate([s.test for s in splits if s.repeat == rep])
        assert sorted(tests.tolist()) == list(range(n))  # every subject tested once
    for s in splits:
        assert_disjoint(s)
        assert len(s.train) + len(s.val) + len(s.test) == n
        # stratification: class balance of each partition close to overall
        for part in (s.train, s.val, s.test):
            assert abs(y[part].mean() - y.mean()) < 0.2


def test_nested_splits_are_reproducible():
    y = [0, 1] * 20
    a = [s.test.tolist() for s in nested_splits(y, 5, seed=7)]
    b = [s.test.tolist() for s in nested_splits(y, 5, seed=7)]
    c = [s.test.tolist() for s in nested_splits(y, 5, seed=8)]
    assert a == b and a != c


def test_assert_disjoint_detects_leakage(records):
    bad = Split(0, 0, np.array([0, 1, 2]), np.array([3]), np.array([2, 4]))
    with pytest.raises(AssertionError):
        assert_disjoint(bad, records)


def test_invalid_inner_frac():
    with pytest.raises(ValueError):
        list(nested_splits([0, 1] * 10, 2, inner_frac=1.5))
