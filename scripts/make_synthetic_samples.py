"""Write the synthetic validation samples to disk so they can be viewed.

``validate_metaqc.py`` builds these in a temp dir and deletes them; this script
writes them to a folder you choose and prints the known ground-truth values, so
you can open each volume in a viewer and confirm by eye that it matches the
numbers ``metaqc`` reports.

Run:
    python scripts/make_synthetic_samples.py --out ./synthetic_qc

Then inspect, e.g.:
    python scripts/inspect_nifti.py ./synthetic_qc/sample2_image.nii.gz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import nibabel as nib

from clinmriqc import metaqc


def _save(path, array, spacing=(1.0, 1.0, 1.0)):
    img = nib.Nifti1Image(array.astype(np.float32), np.diag([*spacing, 1.0]))
    img.header.set_zooms(spacing)
    nib.save(img, path)


def main():
    ap = argparse.ArgumentParser(description="Write synthetic QC samples to disk.")
    ap.add_argument("--out", default="./synthetic_qc", help="Output folder.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    shape = (40, 40, 40)

    # Sample 1: uniform cube of 100 inside a matching mask.
    img1 = np.zeros(shape, dtype=np.float32); img1[10:30, 10:30, 10:30] = 100.0
    mask1 = np.zeros(shape, dtype=np.float32); mask1[10:30, 10:30, 10:30] = 1.0
    _save(os.path.join(args.out, "sample1_image.nii.gz"), img1)
    _save(os.path.join(args.out, "sample1_mask.nii.gz"), mask1)

    # Sample 2: 50/150 split inside the mask (mean must be 100).
    img2 = np.zeros(shape, dtype=np.float32)
    img2[10:30, 10:30, 10:20] = 50.0
    img2[10:30, 10:30, 20:30] = 150.0
    mask2 = np.zeros(shape, dtype=np.float32); mask2[10:30, 10:30, 10:30] = 1.0
    _save(os.path.join(args.out, "sample2_image.nii.gz"), img2)
    _save(os.path.join(args.out, "sample2_mask.nii.gz"), mask2)

    # Sample 3: anisotropic spacing 0.8 x 0.8 x 3.0.
    img3 = np.zeros((20, 20, 20), dtype=np.float32); img3[5:15, 5:15, 5:15] = 100.0
    _save(os.path.join(args.out, "sample3_aniso.nii.gz"), img3, spacing=(0.8, 0.8, 3.0))

    # Sample 4: nearly-empty volume (tiny corner) -> should fail foreground.
    img4 = np.zeros(shape, dtype=np.float32); img4[0:5, 0:5, 0:5] = 100.0
    _save(os.path.join(args.out, "sample4_empty.nii.gz"), img4)

    # Sample 5: an off-centre blob -> non-zero centroid offset.
    img5 = np.zeros(shape, dtype=np.float32); img5[2:12, 2:12, 2:12] = 100.0
    mask5 = (img5 > 0).astype(np.float32)
    _save(os.path.join(args.out, "sample5_offcentre_image.nii.gz"), img5)
    _save(os.path.join(args.out, "sample5_offcentre_mask.nii.gz"), mask5)

    # Print the known values + what metaqc computes, side by side.
    print(f"Synthetic samples written to: {os.path.abspath(args.out)}\n")
    print("Known construction vs metaqc output:\n")

    # Full ground-truth description, also written to a text file so a reviewer
    # knows EXACTLY what each volume contains and what the correct numbers are.
    descriptions = {
        "sample1": (
            "Sample 1 - uniform cube.\n"
            "  Volume: 40x40x40 = 64000 voxels, all zero except a 20x20x20 cube\n"
            "          (indices 10:30 on each axis) = 8000 voxels set to 100.\n"
            "  Brain mask: exactly that cube (8000 voxels = 1).\n"
            "  GROUND TRUTH (stats over the mask):\n"
            "    intensity_mean = 100 (every masked voxel is 100)\n"
            "    intensity_std  = 0   (uniform)\n"
            "    foreground_fraction = 8000/64000 = 0.125\n"
            "    centroid_offset ~ 0  (cube is centred in the volume)\n"
        ),
        "sample2": (
            "Sample 2 - two-valued region (the key test for the mask confusion).\n"
            "  Volume: 40x40x40. Inside the 20x20x20 mask, half the slices = 50,\n"
            "          the other half = 150. Outside the mask = 0.\n"
            "  GROUND TRUTH (stats over the mask, of the IMAGE not the mask):\n"
            "    intensity_mean = (50+150)/2 = 100   <-- NOT 1 (the mask value)\n"
            "    intensity_std  = 50   (each half is 50 away from the mean)\n"
            "    centroid_offset ~ 2.5 mm  (the brighter 150 half pulls the\n"
            "          intensity-weighted centre of mass toward it -- so the\n"
            "          centroid is NOT the geometric centre; this is expected.)\n"
        ),
        "sample3": (
            "Sample 3 - anisotropic voxel spacing (metadata QC).\n"
            "  Volume: 20x20x20, voxel spacing set to 0.8 x 0.8 x 3.0 mm.\n"
            "  GROUND TRUTH:\n"
            "    voxel_spacing = [0.8, 0.8, 3.0]\n"
            "    anisotropy ratio = 3.0/0.8 = 3.75 (< default 8 -> passes)\n"
        ),
        "sample4": (
            "Sample 4 - nearly-empty volume (foreground-threshold test).\n"
            "  Volume: 40x40x40, only a 5x5x5 corner (125 voxels) set to 100.\n"
            "  GROUND TRUTH:\n"
            "    foreground_fraction = 125/64000 = 0.00195\n"
            "    foreground check FAILS at default 0.05 floor (volume too empty).\n"
        ),
        "sample5": (
            "Sample 5 - off-centre blob (centroid test).\n"
            "  Volume: 40x40x40, a 10x10x10 cube at indices 2:12 (near a corner).\n"
            "  GROUND TRUTH:\n"
            "    intensity_mean = 100, std = 0 (uniform blob)\n"
            "    centroid_offset LARGE (~22 mm): blob is far from the volume centre.\n"
        ),
    }
    desc_path = os.path.join(args.out, "SAMPLE_DESCRIPTIONS.txt")
    with open(desc_path, "w", encoding="utf-8") as fh:
        fh.write("Synthetic QC samples - ground-truth descriptions\n")
        fh.write("=" * 50 + "\n\n")
        for v in descriptions.values():
            fh.write(v + "\n")
    print(f"Full ground-truth descriptions written to: {desc_path}\n")

    def report(label, img_path, mask_path=None):
        img = metaqc.load_nifti(img_path)
        mask = metaqc.load_nifti(mask_path).astype(bool) if mask_path else None
        f = metaqc.compute_features(img, brain_mask=mask, affine=np.eye(4))
        print(f"{label}")
        print(f"   foreground_method   = {f['foreground_method']}")
        print(f"   foreground_fraction = {f['foreground_fraction']}")
        print(f"   intensity_mean      = {f['intensity_mean']}")
        print(f"   intensity_std       = {f['intensity_std']}")
        print(f"   centroid_offset_mm  = {f['centroid_offset_mm']}\n")

    report("Sample 1 (cube=100 in mask; expect mean 100, std 0, frac 0.125)",
           os.path.join(args.out, "sample1_image.nii.gz"),
           os.path.join(args.out, "sample1_mask.nii.gz"))
    report("Sample 2 (50/150 in mask; expect mean 100, std 50)",
           os.path.join(args.out, "sample2_image.nii.gz"),
           os.path.join(args.out, "sample2_mask.nii.gz"))
    # Sample 3 (anisotropic spacing) - a metadata-QC sample, shown via header.
    m3 = metaqc.check_metadata(os.path.join(args.out, "sample3_aniso.nii.gz"))
    print("Sample 3 (anisotropic spacing; expect voxel_spacing [0.8, 0.8, 3.0])")
    print(f"   voxel_spacing       = {m3['metadata']['voxel_spacing']}")
    print(f"   spacing check       = {m3['checks']['voxel_spacing']['status']}\n")
    # Sample 4 (nearly empty) - a feature-QC threshold sample.
    r4 = metaqc.run_qc(os.path.join(args.out, "sample4_empty.nii.gz"))
    print("Sample 4 (nearly-empty volume; expect foreground check to FAIL)")
    print(f"   foreground_fraction = {r4['features']['foreground_fraction']}")
    print(f"   foreground check    = {r4['feature_qc']['checks']['foreground']['status']}\n")
    report("Sample 5 (off-centre blob; expect non-zero centroid offset)",
           os.path.join(args.out, "sample5_offcentre_image.nii.gz"),
           os.path.join(args.out, "sample5_offcentre_mask.nii.gz"))

    print("Inspect any volume with:")
    print(f"   python scripts/inspect_nifti.py {os.path.join(args.out, 'sample2_image.nii.gz')}")


if __name__ == "__main__":
    main()
