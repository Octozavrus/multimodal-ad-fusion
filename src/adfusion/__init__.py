"""adfusion: preprocessing of paired sMRI and FDG-PET scans for multimodal fusion.

For every subject the FDG-PET volume is co-registered to the subject's own
T1-weighted MRI, both volumes are cropped with one T1-derived bounding box,
resampled to a common shape and intensity-normalised, and the registration
quality is recorded as a Dice overlap. The output is a cache of paired NumPy
arrays and a manifest that downstream models can read.

    preprocessing  the preprocessing chain and the resumable batch runner
    synthetic      synthetic raw scans for tests and demonstrations
    cli            the ``adfusion`` command

No imaging data are distributed with this package.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
