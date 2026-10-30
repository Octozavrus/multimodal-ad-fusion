"""Manifest loading, paired-volume dataset and leakage-safe nested splits.

A manifest is a CSV with the columns ``subject, label, mri, pet``. Each row is
one subject; ``mri`` and ``pet`` point to preprocessed ``.npy`` volumes.

Splitting is done at the subject level and stratified by label. The outer
folds are the test sets. An inner validation split is carved out of each
outer training set only, so the test fold is never used for epoch or
threshold selection.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

LABEL_MAP = {"CN": 0, "AD": 1}
REQUIRED_COLUMNS = ("subject", "label", "mri", "pet")


def load_manifest(path: str | os.PathLike, data_root: str | None = None) -> list[dict]:
    """Read a manifest and keep AD and CN subjects only.

    If ``data_root`` is given, volume paths are re-rooted to that directory
    (useful when a cache is moved between machines).
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"manifest {path} is missing columns: {missing}")
        rows = []
        for r in reader:
            lab = r["label"].strip().upper()
            if lab not in LABEL_MAP:
                continue
            mri, pet = r["mri"], r["pet"]
            if data_root:
                mri = os.path.join(data_root, os.path.basename(mri))
                pet = os.path.join(data_root, os.path.basename(pet))
            rows.append(
                {"subject": r["subject"], "label": lab, "y": LABEL_MAP[lab], "mri": mri, "pet": pet}
            )
    subjects = [r["subject"] for r in rows]
    if len(subjects) != len(set(subjects)):
        raise ValueError("manifest contains duplicate subject IDs")
    return rows


class PairedVolumeDataset(Dataset):
    """Returns ``(mri[1,D,H,W], pet[1,D,H,W], label)``.

    Augmentation applies the same left-right flip to both modalities (so
    their spatial correspondence is kept) and adds small Gaussian noise.
    """

    def __init__(
        self,
        records: Sequence[dict],
        augment: bool = False,
        flip_p: float = 0.5,
        noise_std: float = 0.01,
    ):
        self.records = list(records)
        self.augment = augment
        self.flip_p = flip_p
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, mri: np.ndarray, pet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # The global NumPy RNG is used on purpose: PyTorch re-seeds it in every
        # DataLoader worker, so augmentation differs between workers and epochs
        # while remaining reproducible from the run seed.
        if np.random.rand() < self.flip_p:  # noqa: NPY002
            mri = np.flip(mri, axis=0).copy()
            pet = np.flip(pet, axis=0).copy()
        if self.noise_std > 0:
            mri = mri + np.random.normal(0, self.noise_std, mri.shape)  # noqa: NPY002
            pet = pet + np.random.normal(0, self.noise_std, pet.shape)  # noqa: NPY002
            mri, pet = np.clip(mri, 0, 1), np.clip(pet, 0, 1)
        return mri.astype(np.float32), pet.astype(np.float32)

    def __getitem__(self, i: int):
        r = self.records[i]
        mri = np.load(r["mri"]).astype(np.float32)
        pet = np.load(r["pet"]).astype(np.float32)
        if mri.shape != pet.shape:
            raise ValueError(f"{r['subject']}: MRI {mri.shape} and PET {pet.shape} differ")
        if self.augment:
            mri, pet = self._augment(mri, pet)
        return (
            torch.from_numpy(np.ascontiguousarray(mri))[None],
            torch.from_numpy(np.ascontiguousarray(pet))[None],
            torch.tensor(r["y"], dtype=torch.float32),
        )


def make_loaders(
    train_recs: Sequence[dict],
    eval_recs: Sequence[dict],
    batch_size: int = 8,
    num_workers: int = 0,
    augment: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Training loader (shuffled, optionally augmented) and evaluation loader."""
    pin = torch.cuda.is_available()
    tr = DataLoader(
        PairedVolumeDataset(train_recs, augment=augment),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    ev = DataLoader(
        PairedVolumeDataset(eval_recs, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return tr, ev


@dataclass(frozen=True)
class Split:
    """Index arrays for one outer fold of nested cross-validation."""

    repeat: int
    fold: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def nested_splits(
    labels: Sequence[int],
    n_splits: int = 5,
    inner_frac: float = 0.2,
    seed: int = 42,
    repeats: int = 1,
) -> Iterator[Split]:
    """Yield subject-level stratified nested splits.

    For every repeat ``r`` the outer split uses ``seed + r``. Inside each outer
    training set, one stratified fold of size about ``inner_frac`` becomes the
    inner validation set.
    """
    y = np.asarray(labels)
    if not 0 < inner_frac < 1:
        raise ValueError("inner_frac must be in (0, 1)")
    n_inner = max(2, int(round(1 / inner_frac)))
    for rep in range(repeats):
        outer = StratifiedKFold(n_splits, shuffle=True, random_state=seed + rep)
        for k, (tr_idx, te_idx) in enumerate(outer.split(np.zeros(len(y)), y)):
            inner = StratifiedKFold(n_inner, shuffle=True, random_state=seed + rep)
            tr2, va2 = next(inner.split(np.zeros(len(tr_idx)), y[tr_idx]))
            yield Split(rep, k, tr_idx[tr2], tr_idx[va2], te_idx)


def assert_disjoint(split: Split, records: Sequence[dict] | None = None) -> None:
    """Raise if any index (or subject ID) appears in more than one partition."""
    parts = {"train": split.train, "val": split.val, "test": split.test}
    seen: dict[int, str] = {}
    for name, idx in parts.items():
        for i in idx:
            if int(i) in seen:
                raise AssertionError(f"index {i} in both {seen[int(i)]} and {name}")
            seen[int(i)] = name
    if records is not None:
        subj = {n: {records[i]["subject"] for i in idx} for n, idx in parts.items()}
        tr, va, te = subj["train"], subj["val"], subj["test"]
        if tr & te or va & te or tr & va:
            raise AssertionError("subject leakage between partitions")
