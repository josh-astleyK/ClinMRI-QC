"""Example: how to call metaqc on your own data and SEE the results.

Edit the paths near the top to point at your data, then run:
    python scripts/example_usage.py

This prints the result dictionaries (which run_qc / check_completeness return)
so you can actually read them. Nothing here is special -- it is just the calls
from the README with print()/json around them so the output is visible.
"""

from __future__ import annotations

import glob
import json
import os

from clinmriqc import metaqc

# --------------------------------------------------------------------------- #
# EDIT THESE PATHS to point at one of your real subjects.
# --------------------------------------------------------------------------- #
DATASET = r"C:\Users\rongh\BMEIShack\datasets\open_ms_data\cross_sectional\coregistered"
SUBJECT = "patient01"

subject_dir = os.path.join(DATASET, SUBJECT)
image       = os.path.join(subject_dir, "T1W.nii.gz")
image_ce    = os.path.join(subject_dir, "T1WKS.nii.gz")     # post-contrast T1
brain_mask  = os.path.join(subject_dir, "brainmask.nii.gz")
lesion_mask = os.path.join(subject_dir, "consensus_gt.nii.gz")

# What you EXPECT each subject to contain (drives completeness).
config = {
    "required_modalities": ["T1w", "FLAIR", "T1CE", "T2w"],
    "required_masks": ["brain"],
    "expected_timepoints": None,          # set to an int for longitudinal data
}

# Thresholds you want to apply (override any subset of metaqc.DEFAULT_THRESHOLDS).
thresholds = {
    "max_centroid_offset_mm": 45.0,       # relax the centroid warning
}


def show(title, result):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(json.dumps(result, indent=2, default=str))


# --------------------------------------------------------------------------- #
# 1) ONE IMAGE, no mask
# --------------------------------------------------------------------------- #
show("1) run_qc on ONE image (no mask)",
     metaqc.run_qc(image, thresholds=thresholds))

# --------------------------------------------------------------------------- #
# 2) ONE IMAGE + brain mask (intensity stats restricted to the brain)
# --------------------------------------------------------------------------- #
show("2) run_qc on one image + brain mask",
     metaqc.run_qc(image, brain_mask_path=brain_mask, thresholds=thresholds))

# --------------------------------------------------------------------------- #
# 3) ONE IMAGE + brain mask + lesion mask (post-contrast T1 + both masks)
# --------------------------------------------------------------------------- #
show("3) run_qc on one image + brain mask + lesion mask",
     metaqc.run_qc(image_ce, brain_mask_path=brain_mask,
                   lesion_mask_path=lesion_mask, thresholds=thresholds))

# --------------------------------------------------------------------------- #
# 4) SUBJECT COMPLETENESS: are all expected modalities/masks present?
# --------------------------------------------------------------------------- #
subject_files = sorted(glob.glob(os.path.join(subject_dir, "*.nii*")))
show("4) check_completeness for the whole subject",
     metaqc.check_completeness(subject_files, config))

print("\nDone. Edit DATASET/SUBJECT at the top to try another subject.")
