"""
Debug script — deep analysis of mask and skeleton for trajectory extraction.
Analyzes the binary mask BEFORE skeletonization and compares to skeleton result.
"""
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import (convolve as nd_convolve, label as nd_label,
                           generate_binary_structure, distance_transform_edt)
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).parent))
from data_postprocessing.vtk_analysis import CenterlineExtractor

OUT_DIR = Path(__file__).parent / "test_data" / "debug_output"
OUT_DIR.mkdir(exist_ok=True)

CASES = [
    ("g21.nii.gz", 2),
    ("g40.nii.gz", 1),
    ("p13.nii.gz", 2),
]


def analyze_mask_region(name, binary_zyx, img_sitk):
    """Analyze the binary mask before skeletonization."""
    print(f"\n--- Mask analysis (before skeletonization) ---")
    struct26 = generate_binary_structure(3, 3)

    total_voxels = int(binary_zyx.sum())
    comp_labeled, ncomp = nd_label(binary_zyx, structure=struct26)
    comp_sizes = sorted([int((comp_labeled == i).sum()) for i in range(1, ncomp + 1)], reverse=True)
    print(f"  Total voxels: {total_voxels}")
    print(f"  Connected components in MASK: {ncomp}, sizes: {comp_sizes[:10]}")

    if ncomp > 1:
        print(f"  !! Mask itself is disconnected — gap is in the PREDICTION, not skeletonization")
        # Find gap between the two largest mask components
        if ncomp >= 2:
            # Get component 1 and 2 voxels in world coords
            spacing = np.array(img_sitk.GetSpacing())
            origin = np.array(img_sitk.GetOrigin())
            direction = np.array(img_sitk.GetDirection()).reshape(3, 3)

            def vox_to_world_zyx(idx_zyx):
                # idx_zyx is (z,y,x), sitk uses (x,y,z)
                idx_xyz = idx_zyx[:, ::-1].astype(float)
                return (idx_xyz * spacing) @ direction.T + origin

            sizes_with_idx = sorted(
                [(int((comp_labeled == i).sum()), i) for i in range(1, ncomp + 1)],
                reverse=True
            )
            c1_vox = np.argwhere(comp_labeled == sizes_with_idx[0][1])
            c2_vox = np.argwhere(comp_labeled == sizes_with_idx[1][1])
            w1 = vox_to_world_zyx(c1_vox)
            w2 = vox_to_world_zyx(c2_vox)

            # Min distance between the two components (sample for speed)
            step = max(1, len(w1) // 200)
            dists = np.linalg.norm(w1[::step, None, :] - w2[None, ::step, :], axis=2)
            min_gap_mm = float(dists.min())
            print(f"  Min gap between mask component 1 and 2: {min_gap_mm:.2f} mm")
    else:
        print(f"  Mask is ONE connected region — gap is introduced by skeletonization")
        # Analyze the mask thickness (distance transform)
        # Use the swapped (XYZ) version for consistency with skeleton
        binary_xyz = np.swapaxes(binary_zyx, 0, -1)
        dt = distance_transform_edt(binary_xyz)
        skel_vox = dt[binary_xyz > 0]
        print(f"  Mask thickness (dist transform at mask voxels):")
        print(f"    min={skel_vox.min():.2f}  median={np.median(skel_vox):.2f}  "
              f"max={skel_vox.max():.2f} voxels")
        print(f"  Thin regions (dt < 1.5 voxels): {int((skel_vox < 1.5).sum())} voxels "
              f"({100*(skel_vox < 1.5).mean():.1f}%)")
        print(f"  Very thin (dt < 1.0): {int((skel_vox < 1.0).sum())} voxels "
              f"({100*(skel_vox < 1.0).mean():.1f}%)")


def analyze_skeleton_region(name, label, skeleton_arr, img_sitk):
    """Analyze skeleton components in detail."""
    print(f"\n--- Skeleton analysis ---")
    struct26 = generate_binary_structure(3, 3)
    kernel = np.ones((3, 3, 3), dtype=np.uint8); kernel[1, 1, 1] = 0
    nc = nd_convolve(skeleton_arr.astype(np.uint8), kernel, mode='constant', cval=0)

    comp_labeled, ncomp = nd_label(skeleton_arr, structure=struct26)
    comp_sizes = sorted([(int((comp_labeled == i).sum()), i) for i in range(1, ncomp + 1)],
                        reverse=True)
    print(f"  Total skeleton voxels: {int(skeleton_arr.sum())}")
    print(f"  Connected components: {ncomp}")

    for rank, (size, ci) in enumerate(comp_sizes):
        comp_mask = comp_labeled == ci
        voxels = np.argwhere(comp_mask)
        world_pts = np.array([img_sitk.TransformIndexToPhysicalPoint(tuple(map(int, v)))
                               for v in voxels])
        eps = np.argwhere(comp_mask & (nc == 1))
        ep_world = [img_sitk.TransformIndexToPhysicalPoint(tuple(map(int, e))) for e in eps]
        spans = world_pts.max(0) - world_pts.min(0)
        print(f"\n  Component {rank+1} ({size} voxels):")
        print(f"    Span (mm): X={spans[0]:.1f}  Y={spans[1]:.1f}  Z={spans[2]:.1f}")
        print(f"    Endpoints ({len(eps)}): {[tuple(round(x,1) for x in p) for p in ep_world]}")
        print(f"    Degree dist: " +
              str({int(d): int((nc[comp_mask] == d).sum())
                   for d in sorted(np.unique(nc[comp_mask]))}))

    # Gap between all pairs of components
    if ncomp >= 2:
        print(f"\n  Gaps between component endpoints (mm):")
        for ri, (si, ci) in enumerate(comp_sizes):
            for rj, (sj, cj) in enumerate(comp_sizes):
                if rj <= ri:
                    continue
                eps_i = np.argwhere((comp_labeled == ci) & (nc == 1))
                eps_j = np.argwhere((comp_labeled == cj) & (nc == 1))
                if len(eps_i) == 0 or len(eps_j) == 0:
                    continue
                wi = np.array([img_sitk.TransformIndexToPhysicalPoint(tuple(map(int, e)))
                                for e in eps_i])
                wj = np.array([img_sitk.TransformIndexToPhysicalPoint(tuple(map(int, e)))
                                for e in eps_j])
                dists = np.linalg.norm(wi[:, None, :] - wj[None, :, :], axis=2)
                print(f"    Comp {ri+1} <-> Comp {rj+1}: min gap = {dists.min():.2f} mm")


def plot_skeleton_on_mask(name, label, binary_xyz, skeleton_arr, img_sitk, comp_labeled):
    """Plot skeleton components overlaid on max-projection of the mask."""
    struct26 = generate_binary_structure(3, 3)
    ncomp = comp_labeled.max()
    colors = ['red', 'lime', 'cyan', 'yellow', 'magenta']

    # Max projections along each axis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{name} label={label} — mask (gray) + skeleton components (colored)")

    spacing = np.array(img_sitk.GetSpacing())  # (sx, sy, sz) in XYZ

    proj_configs = [
        (2, 0, 1, "Z-proj (X vs Y)", spacing[0], spacing[1]),
        (1, 0, 2, "Y-proj (X vs Z)", spacing[0], spacing[2]),
        (0, 1, 2, "X-proj (Y vs Z)", spacing[1], spacing[2]),
    ]

    for ax, (proj_ax, xi, yi, title, sx, sy) in zip(axes, proj_configs):
        mask_proj = binary_xyz.max(axis=proj_ax)
        extent = [0, mask_proj.shape[1] * sy, 0, mask_proj.shape[0] * sx]
        ax.imshow(mask_proj.T, origin='lower', cmap='gray', alpha=0.3,
                  aspect='auto', extent=extent)

        for ci in range(1, ncomp + 1):
            comp_mask = comp_labeled == ci
            vox = np.argwhere(comp_mask)  # (N, 3) in XYZ
            if len(vox) == 0:
                continue
            color = colors[(ci - 1) % len(colors)]
            ax.scatter(vox[:, yi] * sy, vox[:, xi] * sx,
                       c=color, s=8, label=f'comp {ci} ({len(vox)}v)', zorder=3)

        ax.set_title(title, fontsize=9)
        ax.set_xlabel(['X', 'Y', 'Z'][yi] + ' (mm)')
        ax.set_ylabel(['X', 'Y', 'Z'][xi] + ' (mm)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = OUT_DIR / f"{name}_label{label}_skeleton_on_mask.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def analyze_case(mask_path, label):
    name = mask_path.stem
    print(f"\n{'='*60}")
    print(f"Case: {name}  Label: {label}")
    print(f"{'='*60}")

    mask_img = sitk.ReadImage(str(mask_path))
    mask_arr = sitk.GetArrayFromImage(mask_img)   # ZYX
    binary_zyx = (mask_arr == label).astype(np.uint8)
    print(f"Label {label} voxel count: {binary_zyx.sum()}")

    if binary_zyx.sum() == 0:
        print("ERROR: label not found"); return

    # 1. Analyze the mask BEFORE skeletonization
    analyze_mask_region(name, binary_zyx, mask_img)

    # 2. Skeletonize (same logic as CenterlineExtractor._load_and_skeleton)
    binary_xyz = np.swapaxes(binary_zyx, 0, -1)
    foreground = np.argwhere(binary_xyz)
    skeleton_arr = np.zeros_like(binary_xyz, dtype=bool)
    if len(foreground) > 0:
        lo = np.maximum(foreground.min(axis=0) - 1, 0)
        hi = np.minimum(foreground.max(axis=0) + 2, np.array(binary_xyz.shape))
        slices = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
        skeleton_arr[slices] = skeletonize(binary_xyz[slices].astype(bool))

    # 3. Analyze skeleton
    struct26 = generate_binary_structure(3, 3)
    comp_labeled, _ = nd_label(skeleton_arr, structure=struct26)
    analyze_skeleton_region(name, label, skeleton_arr, mask_img)

    # 4. Plot skeleton on mask
    plot_skeleton_on_mask(name, label, binary_xyz, skeleton_arr, mask_img, comp_labeled)

    # 5. Final pipeline result
    extractor = CenterlineExtractor(spline_smoothing=0.1)
    result = extractor.run(binary_zyx.astype(np.float32), mask_img)
    print(f"\n  Final VTK curve points: {result.GetNumberOfPoints()}")


def main():
    test_dir = Path(__file__).parent / "test_data"
    for filename, label in CASES:
        path = test_dir / filename
        if path.exists():
            analyze_case(path, label)
        else:
            print(f"\nNot found: {path}")
    print(f"\nPlots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
