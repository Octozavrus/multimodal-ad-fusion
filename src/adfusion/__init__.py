"""adfusion: a reproducible sMRI + FDG-PET fusion pipeline for Alzheimer's disease.

The package is organised as a pipes-and-filters pipeline. Each stage is a
separate module with a narrow, testable interface:

    cohort         select subjects that have both modalities (ADNI search exports)
    preprocessing  co-registration, shared crop, resampling, normalisation, QC
    variants       derived input caches (ROI crop, modality-specific inputs)
    data           manifest loading, paired dataset, leakage-safe nested splits
    models         3D CNN encoder and seven fusion configurations
    metrics        AUC, bootstrap intervals, paired tests, Holm correction
    train          nested cross-validation runner
    figures        greyscale publication figures
    synthetic      synthetic raw scans and cohorts for tests and demonstrations
    cli            the ``adfusion`` command

Preprocessing needs only NumPy, SciPy, SimpleITK and pandas. The training and
evaluation stages need the ``train`` extra (PyTorch, scikit-learn, matplotlib).
No imaging data are distributed with this package.
"""

__version__ = "1.1.0"

__all__ = ["__version__"]
