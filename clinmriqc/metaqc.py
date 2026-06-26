"""Metadata and per-sample image feature QC for clinical MRI.

This is the metadata/feature QC contribution to ClinMRI-QC, written as a single
flat module to match the team's per-member layout (cf. ``artifacts.py``,
``coreg.py``, ``contrast.py``). It is **per-sample only**: every function takes a
single image (and optionally a brain mask) and returns a plain dict of named,
documented numbers plus pass/warning/fail flags. There is no cohort/cross-subject
logic here -- that is deliberately out of scope for the team submission.

Two kinds of QC are provided:

1. **Header/metadata QC** (``check_metadata``), reads only the NIfTI
   header: shape, voxel spacing, affine validity, orientation. Answers "is this
   file geometrically valid and plausible?"

2. **Per-sample image-feature QC** (``compute_features`` + ``check_features``) 
   reads voxels: foreground fraction, intensity statistics, centroid offset.
   Answers "are the voxel contents plausible (not empty, not constant, brain
   roughly centred)?"

WHAT EACH NUMBER MEANS (the team asked for this to be explicit)
---------------------------------------------------------------
All intensity statistics are computed over a **foreground voxel set**, never the
whole volume, because MRI background is a large near-zero region that would
dominate any whole volume statistic. How the foreground is chosen depends on
what you pass:

* If you pass a ``brain_mask``: foreground = voxels inside the brain mask, and
  intensity stats are of the **underlying image** within that mask. This is the
  correct, trustworthy option for brain MRI.
* If you pass no mask: foreground is estimated as voxels above a low percentile
  (default 10th) of the volume's own intensity range. The method used is returned
  in ``foreground_method`` so the number is auditable.

So ``intensity_mean`` is the mean intensity of the *image* over the foreground
voxels. It is NOT the mean of mask values -- a binary mask is all 1s, so its
mean would be a meaningless 1.0. (That earlier confusion is why this module
requires the image separately from the mask: you always get image statistics,
restricted to the mask region, never statistics of the mask itself.)

``foreground_fraction`` = (foreground voxel count) / (total voxel count): the
proportion of the volume that is signal rather than background. A tiny value
(e.g. < 0.05) usually means a nearly-empty volume or a failed acquisition.

``centroid_offset_mm`` = distance, in millimetres, between the
intensity-weighted centre of the foreground and the geometric centre of the
volume. Large offsets can indicate the brain is far off-centre (cropping,
mispositioning). It is a soft descriptor, not a hard verdict.

CONFIGURABLE THRESHOLDS
-----------------------
``check_features`` takes a ``thresholds`` dict so callers decide what counts as
a problem, e.g. ``{"min_foreground_fraction": 0.10}`` flags any volume whose
foreground is under 10%. Defaults are in ``DEFAULT_THRESHOLDS``; pass overrides
for any subset.

Every check returns a dict with ``status`` in {"pass","warning","fail"} and a
human-readable ``message``, so results compose into the team's merged report.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel

try:  # Prefer the shared loader if present (team convention).
    from clinmriqc.general import load_nifti
except Exception:  # pragma: no cover - fallback so the file is standalone
    def load_nifti(path: str) -> np.ndarray:
        return np.asarray(nibabel.load(path).dataobj, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Configurable thresholds. Callers override any subset via the `thresholds` arg.
# --------------------------------------------------------------------------- #
DEFAULT_THRESHOLDS: Dict[str, float] = {
    # Geometry (metadata QC)
    "min_voxel_size_mm": 0.1,       # below this, spacing is implausible
    "max_voxel_size_mm": 6.0,       # above this, spacing is implausibly coarse
    "max_anisotropy_ratio": 8.0,    # max(spacing)/min(spacing) above -> warn
    # Image features (per-sample QC)
    "min_foreground_fraction": 0.05,   # below -> likely empty/failed volume (fail)
    "warn_foreground_fraction": 0.10,  # below -> suspicious (warning)
    "min_intensity_std": 1e-6,         # ~0 -> constant image (fail)
    "max_centroid_offset_mm": 30.0,    # above -> brain far off-centre (warning)
}

_FOREGROUND_PERCENTILE = 10.0  # background-removal percentile when no mask given


# --------------------------------------------------------------------------- #
# Metadata (header-only) QC
# --------------------------------------------------------------------------- #
def extract_metadata(path: str) -> dict:
    """Read geometry from a NIfTI header without loading the full voxel array.

    Returns a dict with: shape, n_dims, voxel_spacing (mm, per axis),
    n_volumes (4th dim or 1), orientation (e.g. 'RAS'), affine (4x4 list),
    dtype, and read_ok/error. Never raises.
    """
    try:
        img = nibabel.load(path)
        hdr = img.header
        shape = tuple(int(s) for s in img.shape)
        zooms = tuple(float(z) for z in hdr.get_zooms())
        affine = np.asarray(img.affine, dtype=float)
        try:
            orientation = "".join(nibabel.aff2axcodes(affine))
        except Exception:
            orientation = None
        return {
            "read_ok": True,
            "shape": list(shape),
            "n_dims": len(shape),
            "voxel_spacing": [round(z, 4) for z in zooms[:3]],
            "n_volumes": int(shape[3]) if len(shape) > 3 else 1,
            "orientation": orientation,
            "affine": affine.tolist(),
            "dtype": str(hdr.get_data_dtype()),
            "error": None,
        }
    except Exception as exc:
        return {"read_ok": False, "shape": None, "n_dims": None,
                "voxel_spacing": None, "n_volumes": None, "orientation": None,
                "affine": None, "dtype": None, "error": f"{exc!r}"}


def check_metadata(path: str, thresholds: Optional[dict] = None) -> dict:
    """Header-only QC for a single NIfTI file.

    Returns a dict of named sub-checks, each {"status","message"}, plus an
    overall "status" (worst of the sub-checks) and the raw "metadata".
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    meta = extract_metadata(path)
    checks: Dict[str, dict] = {}

    if not meta["read_ok"]:
        return {"status": "fail", "metadata": meta,
                "checks": {"readable": {"status": "fail",
                                        "message": f"Could not read header: {meta['error']}"}}}

    checks["readable"] = {"status": "pass", "message": "Header read successfully."}

    # Dimensionality: expect 3D (or 4D with a volume axis).
    nd = meta["n_dims"]
    if nd in (3, 4):
        checks["dimensionality"] = {"status": "pass", "message": f"{nd}D image."}
    else:
        checks["dimensionality"] = {"status": "fail",
                                    "message": f"Unexpected dimensionality: {nd}D."}

    # Voxel spacing: present, positive, plausible, not too anisotropic.
    sp = meta["voxel_spacing"] or []
    if not sp or any(s is None for s in sp):
        checks["voxel_spacing"] = {"status": "fail", "message": "Missing voxel spacing."}
    elif any(s <= 0 for s in sp):
        checks["voxel_spacing"] = {"status": "fail",
                                   "message": f"Non-positive voxel spacing: {sp}."}
    elif any(s < t["min_voxel_size_mm"] or s > t["max_voxel_size_mm"] for s in sp):
        checks["voxel_spacing"] = {"status": "warning",
                                   "message": f"Implausible voxel spacing {sp} mm "
                                              f"(expected {t['min_voxel_size_mm']}–{t['max_voxel_size_mm']} mm)."}
    else:
        ratio = max(sp) / min(sp) if min(sp) > 0 else float("inf")
        if ratio > t["max_anisotropy_ratio"]:
            checks["voxel_spacing"] = {"status": "warning",
                                       "message": f"Highly anisotropic voxels (ratio {ratio:.1f}): {sp} mm."}
        else:
            checks["voxel_spacing"] = {"status": "pass",
                                       "message": f"Voxel spacing {sp} mm."}

    # Affine validity: present and non-singular.
    aff = meta["affine"]
    if aff is None:
        checks["affine"] = {"status": "fail", "message": "Missing affine."}
    else:
        det = float(np.linalg.det(np.asarray(aff)[:3, :3]))
        if abs(det) < 1e-8:
            checks["affine"] = {"status": "fail",
                                "message": f"Singular affine (det={det:.2e}); geometry undefined."}
        else:
            checks["affine"] = {"status": "pass",
                                "message": f"Affine valid (det={det:.3g}), orientation {meta['orientation']}."}

    overall = _worst([c["status"] for c in checks.values()])
    return {"status": overall, "metadata": meta, "checks": checks}


# --------------------------------------------------------------------------- #
# Per-sample image-feature QC
# --------------------------------------------------------------------------- #
def _foreground_from_image(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """Estimate foreground when no brain mask is supplied.

    Rule: voxels above the 10th percentile of the volume's own intensity range.
    MRI background is a large near-zero peak, so this removes it without
    assuming intensity units. The method string is returned for auditability.
    """
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=bool), "none(no finite voxels)"
    thr = float(np.percentile(finite, _FOREGROUND_PERCENTILE))
    if thr <= float(finite.min()):
        thr = float(finite.min())
        return image > thr, f"greater-than-min({thr:.4g})"
    return image > thr, f"percentile>{_FOREGROUND_PERCENTILE:g}"


def compute_features(
    image: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
) -> dict:
    """Compute per-sample image features over a foreground voxel set.

    Parameters
    ----------
    image : 3-D float array of voxel intensities.
    brain_mask : optional boolean array, same shape as image. If given,
        statistics are computed over the IMAGE intensities INSIDE this mask
        (the correct option for brain MRI). If omitted, foreground is estimated
        from the image itself (see ``_foreground_from_image``).
    affine : optional 4x4 array. If given, the centroid offset is reported in
        millimetres; otherwise it is reported in voxels.

    Returns
    -------
    dict with (all over the foreground voxel set):
        foreground_method      : str, how foreground was chosen
        foreground_voxels      : int, number of foreground voxels
        total_voxels           : int, total voxels in the volume
        foreground_fraction    : float, foreground_voxels / total_voxels
        intensity_mean/std/min/max/p50/p99 : float, of the IMAGE over foreground
        centroid_offset_mm     : float, distance from foreground centre of mass
                                 to the volume's geometric centre (mm if affine
                                 given, else voxels; key name kept for stability)
        centroid_units         : "mm" or "voxels"
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim > 3:
        image = image[..., 0]
    total = int(image.size)

    if brain_mask is not None:
        mask = np.asarray(brain_mask, dtype=bool)
        if mask.shape != image.shape:
            raise ValueError(f"brain_mask shape {mask.shape} != image shape {image.shape}")
        fg = mask
        method = "brain_mask"
    else:
        fg, method = _foreground_from_image(image)

    fg_values = image[fg]
    fg_count = int(np.count_nonzero(fg))

    if fg_values.size > 0:
        i_mean = float(np.mean(fg_values))
        i_std = float(np.std(fg_values))
        i_min = float(np.min(fg_values))
        i_max = float(np.max(fg_values))
        i_p50 = float(np.percentile(fg_values, 50))
        i_p99 = float(np.percentile(fg_values, 99))
    else:
        i_mean = i_std = i_min = i_max = i_p50 = i_p99 = None

    # Intensity-weighted centroid of the foreground, then offset from the
    # volume's geometric centre.
    centroid_offset = None
    units = "voxels"
    if fg_count > 0:
        weighted = np.where(fg, image, 0.0)
        total_w = float(weighted.sum())
        if total_w > 0:
            grids = np.meshgrid(*[np.arange(s) for s in image.shape], indexing="ij")
            com_vox = np.array([float((g * weighted).sum() / total_w) for g in grids])
            geom_vox = np.array([(s - 1) / 2.0 for s in image.shape])
            if affine is not None:
                aff = np.asarray(affine, dtype=float)
                com_world = aff @ np.array([*com_vox, 1.0])
                geom_world = aff @ np.array([*geom_vox, 1.0])
                centroid_offset = float(np.linalg.norm(com_world[:3] - geom_world[:3]))
                units = "mm"
            else:
                centroid_offset = float(np.linalg.norm(com_vox - geom_vox))
                units = "voxels"

    return {
        "foreground_method": method,
        "foreground_voxels": fg_count,
        "total_voxels": total,
        "foreground_fraction": (fg_count / total) if total else None,
        "intensity_mean": _round(i_mean),
        "intensity_std": _round(i_std),
        "intensity_min": _round(i_min),
        "intensity_max": _round(i_max),
        "intensity_p50": _round(i_p50),
        "intensity_p99": _round(i_p99),
        "centroid_offset_mm": _round(centroid_offset),
        "centroid_units": units,
    }


def check_features(features: dict, thresholds: Optional[dict] = None) -> dict:
    """Grade per-sample features against configurable thresholds.

    Parameters
    ----------
    features : output of ``compute_features``.
    thresholds : optional overrides for ``DEFAULT_THRESHOLDS``. Relevant keys:
        min_foreground_fraction, warn_foreground_fraction, min_intensity_std,
        max_centroid_offset_mm.

    Returns
    -------
    dict of named sub-checks ({"status","message"}) plus overall "status".
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    checks: Dict[str, dict] = {}

    # Foreground fraction.
    ff = features.get("foreground_fraction")
    if ff is None:
        checks["foreground"] = {"status": "fail", "message": "No foreground computed."}
    elif ff < t["min_foreground_fraction"]:
        checks["foreground"] = {"status": "fail",
                                "message": f"Foreground fraction {ff:.3f} below "
                                           f"{t['min_foreground_fraction']:.2f}: likely empty/failed volume."}
    elif ff < t["warn_foreground_fraction"]:
        checks["foreground"] = {"status": "warning",
                                "message": f"Low foreground fraction {ff:.3f} "
                                           f"(< {t['warn_foreground_fraction']:.2f})."}
    else:
        checks["foreground"] = {"status": "pass",
                                "message": f"Foreground fraction {ff:.3f}."}

    # Intensity dynamic range (constant image is a failure).
    istd = features.get("intensity_std")
    if istd is None:
        checks["intensity"] = {"status": "fail", "message": "No intensity statistics."}
    elif istd < t["min_intensity_std"]:
        checks["intensity"] = {"status": "fail",
                               "message": f"Near-constant intensity (std={istd:.2e}): no contrast."}
    else:
        checks["intensity"] = {"status": "pass",
                               "message": f"Intensity mean={features.get('intensity_mean')}, "
                                          f"std={istd}."}

    # Centroid offset (soft warning only).
    off = features.get("centroid_offset_mm")
    if off is None:
        checks["centroid"] = {"status": "warning", "message": "Centroid not computed."}
    elif features.get("centroid_units") == "mm" and off > t["max_centroid_offset_mm"]:
        checks["centroid"] = {"status": "warning",
                              "message": f"Centroid {off:.1f} mm from centre "
                                         f"(> {t['max_centroid_offset_mm']:.0f} mm): brain off-centre?"}
    else:
        checks["centroid"] = {"status": "pass",
                              "message": f"Centroid offset {off} {features.get('centroid_units')}."}

    return {"status": _worst([c["status"] for c in checks.values()]), "checks": checks}


# --------------------------------------------------------------------------- #
# Top-level per-sample entry point
# --------------------------------------------------------------------------- #
def run_qc(
    image_path: str,
    brain_mask_path: Optional[str] = None,
    thresholds: Optional[dict] = None,
) -> dict:
    """Full per-sample metadata + feature QC for one image.

    Loads the image (and brain mask, if given), runs header QC and image-feature
    QC, and returns one combined dict:

        {
          "image": <path>,
          "metadata_qc": {...},     # from check_metadata
          "feature_qc":  {...},     # from check_features
          "features":    {...},     # raw numbers from compute_features
          "status": "pass"/"warning"/"fail",   # worst of the two QC layers
        }

    Reads only the header for metadata QC; reads voxels for feature QC.
    """
    meta_qc = check_metadata(image_path, thresholds)

    feature_qc: dict = {}
    features: dict = {}
    try:
        image = load_nifti(image_path)
        affine = np.asarray(nibabel.load(image_path).affine, dtype=float)
        mask = None
        if brain_mask_path:
            mask = load_nifti(brain_mask_path).astype(bool)
        features = compute_features(image, brain_mask=mask, affine=affine)
        feature_qc = check_features(features, thresholds)
    except Exception as exc:
        feature_qc = {"status": "fail",
                      "checks": {"load": {"status": "fail",
                                          "message": f"Feature QC failed: {exc!r}"}}}

    overall = _worst([meta_qc.get("status", "fail"), feature_qc.get("status", "fail")])
    return {
        "image": image_path,
        "metadata_qc": meta_qc,
        "feature_qc": feature_qc,
        "features": features,
        "status": overall,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_STATUS_RANK = {"pass": 0, "warning": 1, "fail": 2, "unknown": 1}


def _worst(statuses: List[str]) -> str:
    """Worst status among a list, by rank fail > warning > pass."""
    if not statuses:
        return "unknown"
    return max(statuses, key=lambda s: _STATUS_RANK.get(s, 1))


def _round(x: Optional[float], n: int = 4) -> Optional[float]:
    """Round, mapping non-finite to None (JSON-safe)."""
    if x is None:
        return None
    xf = float(x)
    if not np.isfinite(xf):
        return None
    return round(xf, n)


# --------------------------------------------------------------------------- #
# CLI (matches the team's argparse + JSON-output convention)
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="ClinMRI-QC: per-sample metadata + image-feature QC")
    parser.add_argument("--image", required=True, help="Path to a NIfTI image")
    parser.add_argument("--brain_mask", default=None,
                        help="Optional brain mask NIfTI (enables mask-restricted stats)")
    parser.add_argument("--min_foreground_fraction", type=float, default=None,
                        help="Override: flag fail below this foreground fraction")
    parser.add_argument("--outfile", default=None, help="Optional path to save JSON")
    args = parser.parse_args()

    thresholds = {}
    if args.min_foreground_fraction is not None:
        thresholds["min_foreground_fraction"] = args.min_foreground_fraction

    result = run_qc(args.image, brain_mask_path=args.brain_mask, thresholds=thresholds)
    output = json.dumps(result, indent=2)
    print(output)
    if args.outfile:
        with open(args.outfile, "w") as f:
            f.write(output)
        print(f"\nResults saved to {args.outfile}")


if __name__ == "__main__":
    main()
