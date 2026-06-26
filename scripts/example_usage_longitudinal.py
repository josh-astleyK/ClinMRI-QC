"""Example: metaqc on LONGITUDINAL data (open_ms_data longitudinal layout).

Layout per subject (from the dataset README):
    patientX/
      brainmask.nii.gz
      gt.nii.gz                 (lesion change mask)
      study1_T1W.nii.gz  study1_FLAIR.nii.gz  study1_T2W.nii.gz
      study2_T1W.nii.gz  study2_FLAIR.nii.gz  study2_T2W.nii.gz

Note vs the cross-sectional set: timepoints are study1/study2, there is NO
post-contrast (T1CE), and the lesion mask is just gt.nii.gz. The config below
reflects that -- the SAME functions, different configuration.

Edit DATASET/SUBJECT, then:
    python scripts/example_usage_longitudinal.py
"""

from __future__ import annotations

import glob
import json
import os

from clinmriqc import metaqc

# --------------------------------------------------------------------------- #
# EDIT THESE PATHS.
# --------------------------------------------------------------------------- #
DATASET = r"C:\Users\rongh\BMEIShack\datasets\open_ms_data\longitudinal\coregistered"
SUBJECT = "patient01"

subject_dir = os.path.join(DATASET, SUBJECT)
image_s1    = os.path.join(subject_dir, "study1_T1W.nii.gz")    # one timepoint's T1
image_s2    = os.path.join(subject_dir, "study2_FLAIR.nii.gz")  # other timepoint
brain_mask  = os.path.join(subject_dir, "brainmask.nii.gz")
lesion_mask = os.path.join(subject_dir, "gt.nii.gz")

# Longitudinal expectation config: study1/study2 timepoints, T1/FLAIR/T2, no T1CE.
config = {
    "required_modalities": ["T1w", "FLAIR", "T2w"],
    "required_masks": ["brain", "lesion"],
    "timepoint_patterns": ["study"],     # detects study1, study2
    "expected_timepoints": 2,            # imaged twice
}

thresholds = {"max_centroid_offset_mm": 45.0}


def show(title, result):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(json.dumps(result, indent=2, default=str))


# 1) ONE IMAGE (study1 T1), no mask.
show("1) run_qc on ONE image (study1 T1, no mask)",
     metaqc.run_qc(image_s1, thresholds=thresholds))

# 2) ONE IMAGE + brain mask (the brain mask is shared across both studies).
show("2) run_qc on one image + brain mask",
     metaqc.run_qc(image_s1, brain_mask_path=brain_mask, thresholds=thresholds))

# 3) ONE IMAGE (other timepoint) + brain mask + lesion mask.
show("3) run_qc on study2 FLAIR + brain mask + lesion mask",
     metaqc.run_qc(image_s2, brain_mask_path=brain_mask,
                   lesion_mask_path=lesion_mask, thresholds=thresholds))

# 4) SUBJECT COMPLETENESS across BOTH timepoints (the longitudinal check).
subject_files = sorted(glob.glob(os.path.join(subject_dir, "*.nii*")))
show("4) check_completeness across both studies",
     metaqc.check_completeness(subject_files, config))

print("\nDone. Edit DATASET/SUBJECT at the top to try another subject.")
