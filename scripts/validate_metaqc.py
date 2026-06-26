"""Validate metaqc.py numbers against synthetic data with known ground truth.

Build small volumes whose exact statistics are known by construction, then
check that compute_features and check_metadata report those numbers. If the
code is correct, the printed "computed" values match the "expected" values.

Run:  python validate_metaqc.py
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import nibabel as nib

from clinmriqc import metaqc


def _save(path, array, spacing=(1.0, 1.0, 1.0)):
    img = nib.Nifti1Image(array.astype(np.float32), np.diag([*spacing, 1.0]))
    img.header.set_zooms(spacing)
    nib.save(img, path)


def case(name, expected, computed, tol=1e-3):
    ok = abs(expected - computed) <= tol if computed is not None else False
    flag = "OK " if ok else "XX "
    print(f"  [{flag}] {name}: expected {expected}, computed {computed}")
    return ok


def main():
    tmp = tempfile.mkdtemp(prefix="validate_metaqc_")
    all_ok = True
    print("Validating metaqc.py against synthetic ground truth\n")

    # ----------------------------------------------------------------- #
    # Sample 1: a cube of known intensity inside a known brain mask.
    # 40x40x40 = 64000 voxels. A 20x20x20 cube (8000 voxels) of value 100,
    # inside a brain mask that is exactly that cube. Background = 0.
    # ----------------------------------------------------------------- #
    print("Sample 1 - uniform cube of intensity 100 inside a matching brain mask:")
    shape = (40, 40, 40)
    img = np.zeros(shape, dtype=np.float32)
    img[10:30, 10:30, 10:30] = 100.0
    mask = np.zeros(shape, dtype=np.float32)
    mask[10:30, 10:30, 10:30] = 1.0
    img_p = os.path.join(tmp, "s1_img.nii.gz")
    mask_p = os.path.join(tmp, "s1_mask.nii.gz")
    _save(img_p, img); _save(mask_p, mask)

    feats = metaqc.compute_features(
        metaqc.load_nifti(img_p),
        brain_mask=metaqc.load_nifti(mask_p).astype(bool),
        affine=np.eye(4),
    )
    # Ground truth: inside the mask every voxel is 100, so mean=100, std=0,
    # min=max=100. Foreground voxels = 8000. Fraction = 8000/64000 = 0.125.
    all_ok &= case("intensity_mean (mask-restricted)", 100.0, feats["intensity_mean"])
    all_ok &= case("intensity_std (uniform -> 0)", 0.0, feats["intensity_std"])
    all_ok &= case("foreground_voxels", 8000, feats["foreground_voxels"])
    all_ok &= case("foreground_fraction (8000/64000)", 0.125, feats["foreground_fraction"])
    # Cube is centred in the volume, so centroid offset ~ 0.
    all_ok &= case("centroid_offset (centred -> ~0)", 0.0, feats["centroid_offset_mm"], tol=0.5)
    print(f"      foreground_method = {feats['foreground_method']} (should be 'brain_mask')\n")

    # ----------------------------------------------------------------- #
    # Sample 2: two-valued region to check mean/std are of the IMAGE, not mask.
    # Half the mask region = 50, half = 150. Mean should be 100, std = 50.
    # This is the key test for the team's confusion: a binary mask is all 1s,
    # but intensity_mean must reflect the IMAGE (100), never the mask value (1).
    # ----------------------------------------------------------------- #
    print("Sample 2 - image values 50 and 150 inside the mask (mean must be 100, not 1):")
    img2 = np.zeros(shape, dtype=np.float32)
    img2[10:30, 10:30, 10:20] = 50.0
    img2[10:30, 10:30, 20:30] = 150.0
    mask2 = np.zeros(shape, dtype=np.float32)
    mask2[10:30, 10:30, 10:30] = 1.0
    img2_p = os.path.join(tmp, "s2_img.nii.gz")
    mask2_p = os.path.join(tmp, "s2_mask.nii.gz")
    _save(img2_p, img2); _save(mask2_p, mask2)
    feats2 = metaqc.compute_features(
        metaqc.load_nifti(img2_p),
        brain_mask=metaqc.load_nifti(mask2_p).astype(bool),
        affine=np.eye(4),
    )
    # Equal halves of 50 and 150: mean = 100, std = 50.
    all_ok &= case("intensity_mean (image, not mask value)", 100.0, feats2["intensity_mean"])
    all_ok &= case("intensity_std (sqrt of mean sq dev = 50)", 50.0, feats2["intensity_std"])
    print("      ^ proves stats are of the IMAGE within the mask, never the mask's own 1s\n")

    # ----------------------------------------------------------------- #
    # Sample 3: known voxel spacing for metadata QC.
    # ----------------------------------------------------------------- #
    print("Sample 3 - known voxel spacing 0.8 x 0.8 x 3.0 mm (anisotropic):")
    img3 = np.zeros((20, 20, 20), dtype=np.float32); img3[5:15, 5:15, 5:15] = 100
    img3_p = os.path.join(tmp, "s3.nii.gz")
    _save(img3_p, img3, spacing=(0.8, 0.8, 3.0))
    meta = metaqc.check_metadata(img3_p)
    sp = meta["metadata"]["voxel_spacing"]
    print(f"      computed voxel_spacing = {sp} (expected [0.8, 0.8, 3.0])")
    all_ok &= (sp == [0.8, 0.8, 3.0])
    # Anisotropy ratio = 3.0/0.8 = 3.75, below default 8 -> should pass.
    print(f"      voxel_spacing check status = {meta['checks']['voxel_spacing']['status']}\n")

    # ----------------------------------------------------------------- #
    # Sample 4: thresholding behaviour - an almost-empty volume should FAIL
    # the foreground check at the default 5% floor.
    # ----------------------------------------------------------------- #
    print("Sample 4 - nearly-empty volume (should FAIL foreground threshold):")
    img4 = np.zeros((40, 40, 40), dtype=np.float32)
    img4[0:5, 0:5, 0:5] = 100  # 125 voxels = 0.2% -> below 5%
    img4_p = os.path.join(tmp, "s4.nii.gz")
    _save(img4_p, img4)
    res4 = metaqc.run_qc(img4_p)
    ff = res4["features"]["foreground_fraction"]
    status = res4["feature_qc"]["checks"]["foreground"]["status"]
    print(f"      foreground_fraction = {ff:.4f}, foreground check = {status} (expected 'fail')")
    all_ok &= (status == "fail")
    # And confirm a custom threshold works: set min to 0.001 -> should pass now.
    res4b = metaqc.run_qc(img4_p, thresholds={"min_foreground_fraction": 0.001,
                                              "warn_foreground_fraction": 0.001})
    status_b = res4b["feature_qc"]["checks"]["foreground"]["status"]
    print(f"      with min_foreground_fraction=0.001 -> {status_b} (expected 'pass')")
    all_ok &= (status_b == "pass")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
