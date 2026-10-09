import json

import numpy as np
import pytest

from adfusion import preprocessing as P


def sphere(shape=(40, 40, 40), r=0.6, centre=(0.0, 0.0, 0.0)):
    g = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    d2 = sum((x - c) ** 2 for x, c in zip(g, centre))
    return (d2 <= r**2).astype(np.float32)


class TestBoundingBox:
    def test_box_encloses_foreground(self):
        a = np.zeros((20, 20, 20), np.float32)
        a[5:12, 3:9, 10:15] = 1.0
        bb = P.brain_bbox(a)
        assert bb == (slice(5, 12), slice(3, 9), slice(10, 15))

    def test_empty_volume_returns_full_extent(self):
        a = np.zeros((4, 5, 6), np.float32)
        assert P.brain_bbox(a) == (slice(0, 4), slice(0, 5), slice(0, 6))


class TestResize:
    @pytest.mark.parametrize("src", [(17, 23, 31), (96, 96, 96), (100, 50, 75)])
    def test_exact_target_shape(self, src):
        out = P.resize_to(np.random.default_rng(0).random(src, dtype=np.float32), (32, 32, 32))
        assert out.shape == (32, 32, 32)
        assert out.dtype == np.float32

    def test_constant_volume_stays_constant(self):
        out = P.resize_to(np.full((10, 12, 14), 3.0, np.float32), (20, 20, 20))
        np.testing.assert_allclose(out, 3.0, atol=1e-5)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            P.resize_to(np.zeros((4, 4)), (4, 4, 4))


class TestNormalize:
    def test_range_is_unit_interval(self):
        a = sphere() * np.random.default_rng(0).gamma(2.0, 50.0, (40, 40, 40)).astype(np.float32)
        out = P.normalize(a)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert out.max() > 0.99

    def test_outliers_are_clipped(self):
        a = sphere().copy()
        a[20, 20, 20] = 1e6
        out = P.normalize(a, clip_sigma=5.0)
        assert np.isfinite(out).all()

    def test_all_zero_volume_is_returned_unchanged(self):
        out = P.normalize(np.zeros((5, 5, 5), np.float32))
        assert not out.any()


class TestDice:
    def test_identical_is_one(self):
        s = sphere()
        assert P.dice(s, s) == pytest.approx(1.0)

    def test_disjoint_is_zero(self):
        a, b = np.zeros((10, 10, 10)), np.zeros((10, 10, 10))
        a[:3], b[-3:] = 1, 1
        assert P.dice(a, b) == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            P.dice(np.ones((3, 3, 3)), np.ones((3, 3, 4)))


class TestRoiCrop:
    def test_roi_shape(self):
        out = P.roi_crop(np.zeros((96, 96, 96)), frac=0.5, z_lo=0.25, z_hi=0.75)
        assert out.shape == (48, 48, 48)

    def test_invalid_fractions(self):
        with pytest.raises(ValueError):
            P.roi_crop(np.zeros((8, 8, 8)), z_lo=0.8, z_hi=0.2)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("ADNI/000_S_0001/Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution/x.dcm", "PET"),
        ("ADNI/000_S_0001/MPR__GradWarp__B1_Correction__N3__Scaled/x.nii", "MRI"),
        ("raw/subject_pet.nii.gz", "PET"),
        ("raw/subject_mri.nii.gz", "MRI"),
        ("raw/unknown.nii", None),
    ],
)
def test_classify_modality(path, expected):
    assert P.classify_modality(path) == expected


def test_build_file_index_nifti_and_dicom(tmp_path):
    mri_dir = tmp_path / "000_S_0001" / "MPR_GradWarp_N3_Scaled"
    pet_dir = tmp_path / "000_S_0001" / "Coreg_Avg_Uniform_Resolution"
    mri_dir.mkdir(parents=True)
    pet_dir.mkdir(parents=True)
    (mri_dir / "t1.nii.gz").write_bytes(b"")
    (pet_dir / "slice001.dcm").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignored")
    idx = P.build_file_index(tmp_path)
    assert set(idx) == {"000_S_0001"}
    assert idx["000_S_0001"]["MRI"].endswith("t1.nii.gz")
    assert idx["000_S_0001"]["PET"] == str(pet_dir)


def test_read_labels_first_occurrence_wins(tmp_path):
    f1 = tmp_path / "a.csv"
    f1.write_text("Subject,Group\n000_S_0001,AD\n000_S_0002,cn\n")
    f2 = tmp_path / "b.csv"
    f2.write_text("Subject ID,Research Group\n000_S_0001,CN\n000_S_0003,MCI\n")
    labels = P.read_labels([f1, f2, tmp_path / "missing.csv"])
    assert labels == {"000_S_0001": "AD", "000_S_0002": "CN", "000_S_0003": "MCI"}


# ---------------------------------------------------------------- SimpleITK
sitk = pytest.importorskip("SimpleITK")


def _write_nifti(arr, path, spacing=(2.0, 2.0, 2.0)):
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(path))


def test_registration_recovers_translation(tmp_path):
    fixed_a = sphere((40, 40, 40), r=0.5) + 0.5 * sphere((40, 40, 40), r=0.2, centre=(0.2, 0, 0))
    moving_a = np.roll(fixed_a, shift=(4, -3, 2), axis=(0, 1, 2))
    fixed = sitk.GetImageFromArray(fixed_a)
    moving = sitk.GetImageFromArray(moving_a)
    before = P.dice(fixed_a, moving_a, 0.3)
    out = P.to_array(P.register(moving, fixed, affine=False, iterations=100))
    after = P.dice(fixed_a, out, 0.3)
    assert after > before
    assert after > 0.9


def test_process_subject_and_resumable_run(tmp_path):
    raw = tmp_path / "raw"
    mdir = raw / "000_S_0001" / "MPRAGE"
    pdir = raw / "000_S_0001" / "Coreg_FDG"
    mdir.mkdir(parents=True)
    pdir.mkdir(parents=True)
    t1 = sphere((40, 40, 40), r=0.6) * 100
    pet = np.roll(sphere((40, 40, 40), r=0.55) * 40, 3, axis=0)
    _write_nifti(t1, mdir / "t1.nii.gz")
    _write_nifti(pet, pdir / "pet.nii.gz")
    labels = tmp_path / "labels.csv"
    labels.write_text("Subject,Group\n000_S_0001,AD\n")

    cfg = P.PreprocessConfig(
        download_dir=raw,
        output_dir=tmp_path / "cache",
        label_files=(labels,),
        target_shape=(24, 24, 24),
        affine=False,
    )
    s1 = P.run_preprocessing(cfg)
    assert s1["n_subjects"] == 1 and not s1["failures"]
    m = np.load(tmp_path / "cache" / "000_S_0001_MRI.npy")
    p = np.load(tmp_path / "cache" / "000_S_0001_PET.npy")
    assert m.shape == p.shape == (24, 24, 24)
    assert m.min() >= 0 and m.max() <= 1
    qc = json.loads((tmp_path / "cache" / "000_S_0001_qc.json").read_text())
    assert 0.5 < qc["reg_dice"] <= 1.0

    s2 = P.run_preprocessing(cfg)  # resumed: Dice must survive
    assert s2["dice_mean"] == pytest.approx(round(qc["reg_dice"], 4))


def test_missing_modality_is_reported(tmp_path):
    mdir = tmp_path / "raw" / "000_S_0002" / "MPRAGE"
    mdir.mkdir(parents=True)
    _write_nifti(sphere((20, 20, 20)), mdir / "t1.nii.gz")
    cfg = P.PreprocessConfig(download_dir=tmp_path / "raw", output_dir=tmp_path / "c")
    s = P.run_preprocessing(cfg)
    assert s["n_subjects"] == 0
    assert "missing modality" in s["failures"][0][1]
