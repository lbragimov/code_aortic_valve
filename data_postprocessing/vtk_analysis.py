from pathlib import Path
from skimage.morphology import skeletonize
import SimpleITK as sitk
import numpy as np
from MeshMetrics.utils import vtk_meshing, np2sitk

import vtk


# ------------------
# VTK helpers
# ------------------

def _vtk_polyline_from_points(points):
    """Creates a polygonal chain connecting all points in order."""
    vtk_points = vtk.vtkPoints()
    polyline = vtk.vtkPolyLine()

    polyline.GetPointIds().SetNumberOfIds(len(points))

    for i, p in enumerate(points):
        pid = vtk_points.InsertNextPoint(p)
        polyline.GetPointIds().SetId(i, pid)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    polyline = vtk.vtkPolyData()
    polyline.SetPoints(vtk_points)
    polyline.SetLines(cells)
    return polyline


def _vtk_polyline_to_linesegments(polydata):
    """Create a set of segments from the original polygonal chain."""
    pts = polydata.GetPoints()
    n = pts.GetNumberOfPoints()

    lines = vtk.vtkCellArray()

    for i in range(n - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i + 1)
        lines.InsertNextCell(line)

    out = vtk.vtkPolyData()
    out.SetPoints(pts)
    out.SetLines(lines)
    return out


def _vtk_clip_polyline_with_mesh(polyline, tub_mesh):
    """Trims the polygonal chain so that only the part lying inside the mesh remains."""

    # Create an implicit function from the closed mesh
    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(tub_mesh)

    # Clip polyline with implicit function
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(polyline)
    clipper.SetClipFunction(implicit)
    clipper.InsideOutOn()  # Keep inside
    clipper.Update()

    clipped_polyline = clipper.GetOutput()
    return clipped_polyline


# ------------------
# Main class
# ------------------

class CenterlineExtractor:

    def __init__(self, poly_degree=4, dense_samples=200):
        self.poly_degree = poly_degree
        self.dense_samples = dense_samples

    # ---- Step 1: load and skeletonize ----
    def _load_and_skeleton(self, arr, nii):
        if isinstance(nii, (Path, str)):
            img = sitk.ReadImage(str(nii))
        else:
            img = nii
        arr = arr.astype(bool)
        skeleton_arr = skeletonize(arr)
        skeleton_sitk = np2sitk(skeleton_arr.astype(np.uint8), img.GetSpacing())
        skeleton_sitk.CopyInformation(img)
        return img, skeleton_arr, skeleton_sitk

    # ---- Step 2: collect skeleton points ----
    def _extract_points(self, skeleton_arr, img):
        # extract center points from skeleton image
        center_points = np.argwhere(skeleton_arr > 0)
        # move them to physical space
        center_points_physical = [img.TransformIndexToPhysicalPoint(tuple(map(int, pt))) for pt in center_points]
        return center_points_physical

    # ---- Step 3: polynomial fit ----
    def _fit_curve(self, pts, mesh_polyline=False):
        def _C(t, c):
            out = 0
            for i, ci in enumerate(c):
                out += ci * t ** i
            return out

        N = pts.shape[0]

        d = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
        t = np.concatenate(([0], np.cumsum(d)))
        t = t / t[-1]

        # ---- Build polynomial design matrix ----
        A = np.vstack([np.ones(N), t, t ** 2, t ** 3, t ** 4]).T  # shape (N,5)

        cx, *_ = np.linalg.lstsq(A, pts[:, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(A, pts[:, 1], rcond=None)
        cz, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)

        if mesh_polyline:
            t_dense = np.linspace(-0.5, 1.5, self.dense_samples)
        else:
            t_dense = np.linspace(0.0, 1.0, self.dense_samples)
        curve = np.column_stack([
            _C(t_dense, cx),
            _C(t_dense, cy),
            _C(t_dense, cz)
        ])
        return curve

    # ---- Step 4: clip with mesh ----
    def _clip_with_mesh(self, curve_pts, mesh):
        polyline = _vtk_polyline_from_points(curve_pts)
        clipped = _vtk_clip_polyline_with_mesh(polyline, mesh)
        return clipped

    # ---- Step 5: Full run ----
    def run(self, array_mask, nii_path, mesh_polyline=False):

        array_mask = np.swapaxes(array_mask, 0, -1)

        img, skeleton_arr, skeleton_sitk = self._load_and_skeleton(array_mask, nii_path)
        center_points_physical = self._extract_points(skeleton_arr, img)
        pts = np.asarray(center_points_physical)

        curve = self._fit_curve(pts, mesh_polyline)

        mesh = vtk_meshing(img)
        clipped = self._clip_with_mesh(curve, mesh)
        segments = _vtk_polyline_to_linesegments(clipped)

        return segments