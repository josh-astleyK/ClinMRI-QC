"""Inspect all synthetic QC samples (images and masks) with descriptions.

Companion to make_synthetic_samples.py. It:
  * prints the ground-truth description for every sample, and
  * opens each volume (image or mask) in orthogonal mid-slices so you can see
    exactly what was synthesised.

Run (after writing the samples):
    python scripts/make_synthetic_samples.py --out ./synthetic_qc
    python scripts/inspect_synthetic.py --dir ./synthetic_qc

Just print the descriptions without opening any windows:
    python scripts/inspect_synthetic.py --dir ./synthetic_qc --describe-only

Uses matplotlib (a dev convenience), not part of the metaqc module.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import nibabel as nib

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _mid_slices(vol):
    if vol.ndim > 3:
        vol = vol[..., 0]
    c = [s // 2 for s in vol.shape[:3]]
    return [(vol[c[0], :, :], 0, (1, 2)),
            (vol[:, c[1], :], 1, (0, 2)),
            (vol[:, :, c[2]], 2, (0, 1))]


def _show(path, zooms=None):
    img = nib.load(path)
    vol = np.asarray(img.get_fdata(), dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
    for a, (s, perp, (r, c)) in zip(ax, _mid_slices(vol)):
        aspect = zooms[c] / zooms[r] if zooms[r] else 1.0
        a.imshow(np.rot90(s), cmap="gray", aspect=aspect)
        a.set_title(f"⊥ axis {perp}")
        a.axis("off")
    fig.suptitle(os.path.basename(path))
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Inspect all synthetic QC samples.")
    ap.add_argument("--dir", default="./synthetic_qc", help="Folder of synthetic samples.")
    ap.add_argument("--describe-only", action="store_true",
                    help="Only print descriptions; do not open viewer windows.")
    args = ap.parse_args()

    # Print the descriptions. Try a few encodings: the file may have been
    # written with a non-UTF-8 default on some systems.
    desc = os.path.join(args.dir, "SAMPLE_DESCRIPTIONS.txt")
    if os.path.exists(desc):
        text = None
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                with open(desc, encoding=enc) as fh:
                    text = fh.read()
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            with open(desc, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        print(text)
    else:
        print(f"(no SAMPLE_DESCRIPTIONS.txt in {args.dir}; run make_synthetic_samples.py first)")

    vols = sorted(glob.glob(os.path.join(args.dir, "*.nii.gz")))
    print(f"\nVolumes in {args.dir} ({len(vols)}):")
    for v in vols:
        print(f"  {os.path.basename(v)}")

    if args.describe_only:
        return 0
    if plt is None:
        print("\nmatplotlib not installed; cannot open viewers. "
              "Install with: pip install matplotlib  (or use --describe-only)")
        return 1

    print("\nOpening each volume (close a window to see the next)...")
    for v in vols:
        _show(v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
