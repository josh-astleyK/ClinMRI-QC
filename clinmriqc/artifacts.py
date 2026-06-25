"""Scan-level MRI artifact detection via a pretrained ResNet50 classifier.

Soft-votes a multi-class classifier across sampled axial slices to produce
per-artifact probabilities, then computes analytical image quality metrics.

Example
-------
    import nibabel as nib
    from clinmriqc.artifacts import detect_artifacts
    from clinmriqc.general import load_nifti

    image = load_nifti("T1w.nii.gz")
    mask  = load_nifti("brain_mask.nii.gz").astype(bool)
    result = detect_artifacts(image, mask)

Output schema
-------------
{
    "quality_passed":         bool,
    "artifacts_detected":     ["motion", ...],   # empty if clean
    "artifact_probabilities": {
        "clean":      0.08,
        "motion":     0.71,
        "noise":      0.11,
        "ghosting":   0.03,
        "bias_field": 0.04,
        "gibbs":      0.02,
        "zipper":     0.01,
    },
    "iqms": {
        "motion_blur_score": 0.62,   # EFC — lower = more blur/motion
        "snr":               18.3,   # signal-to-noise ratio
    },
}
"""

import argparse
import json
import numpy as np
from pathlib import Path

from clinmriqc.general import load_nifti
from clinmriqc.iqm.metrics import compute_iqms
from clinmriqc.classifier.model import load_model, predict_volume

_DEFAULT_MODEL = Path(__file__).parent / "classifier" / "best_model.pt"


def detect_artifacts(
    image: np.ndarray,
    brain_mask: np.ndarray,
    model_path: str = None,
    threshold: float = 0.5,
    class_thresholds: dict = None,
    device: str = None,
) -> dict:
    """Detect MRI artifacts and compute image quality metrics.

    Parameters
    ----------
    image            : 3-D float array (H, W, D) — raw voxel intensities.
    brain_mask       : boolean array, same shape as image — brain region only.
    model_path       : path to a trained ResNet50 checkpoint (.pt).
                       Defaults to the bundled best_model.pt.
    threshold        : default probability threshold for flagging an artifact (default 0.5).
    class_thresholds : optional dict of per-class threshold overrides, e.g.
                       {'bias_field': 0.8, 'motion': 0.6}. Overrides the
                       built-in per-class defaults for any class specified.
    device           : 'cpu' or 'cuda'. Auto-detected if not specified.

    Returns
    -------
    dict — see module docstring for full schema.
    """
    model_path = model_path or str(_DEFAULT_MODEL)
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            "Download or train the ResNet50 checkpoint first."
        )

    image      = np.asarray(image, dtype=np.float32)
    brain_mask = np.asarray(brain_mask, dtype=bool)

    iqms   = compute_iqms(image, brain_mask=brain_mask)
    model  = load_model(model_path, arch="resnet50", device=device)
    result = predict_volume(model, image, threshold=threshold,
                            class_thresholds=class_thresholds, device=device)

    return {
        "quality_passed":         result["quality_passed"],
        "artifacts_detected":     result["artifacts_detected"],
        "artifact_probabilities": result["artifact_probabilities"],
        "iqms":                   iqms,
    }


def main():
    parser = argparse.ArgumentParser(description="ClinMRI-QC: detect MRI artifacts")
    parser.add_argument("--image", required=True, help="Path to T1w NIfTI image")
    parser.add_argument("--brain_mask", required=True, help="Path to brain mask NIfTI image")
    parser.add_argument("--model", default=None, help="Path to ResNet50 checkpoint (optional)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Default probability threshold for all artifact classes (default 0.5)")
    parser.add_argument("--threshold_motion",     type=float, default=None)
    parser.add_argument("--threshold_noise",      type=float, default=None)
    parser.add_argument("--threshold_ghosting",   type=float, default=None)
    parser.add_argument("--threshold_bias_field", type=float, default=None)
    parser.add_argument("--threshold_gibbs",      type=float, default=None)
    parser.add_argument("--threshold_zipper",     type=float, default=None)
    parser.add_argument("--outfile", default=None, help="Optional path to save JSON results")
    args = parser.parse_args()

    class_thresholds = {k: v for k, v in {
        'motion':     args.threshold_motion,
        'noise':      args.threshold_noise,
        'ghosting':   args.threshold_ghosting,
        'bias_field': args.threshold_bias_field,
        'gibbs':      args.threshold_gibbs,
        'zipper':     args.threshold_zipper,
    }.items() if v is not None}

    image_arr = load_nifti(args.image)
    mask_arr  = load_nifti(args.brain_mask).astype(bool)

    results = detect_artifacts(image_arr, mask_arr, model_path=args.model,
                               threshold=args.threshold,
                               class_thresholds=class_thresholds or None)
    output = json.dumps(results, indent=2)
    print(output)

    if args.outfile:
        with open(args.outfile, "w") as f:
            f.write(output)
        print(f"\nResults saved to {args.outfile}")
