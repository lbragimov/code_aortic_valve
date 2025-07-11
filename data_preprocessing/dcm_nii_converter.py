from pathlib import Path

import SimpleITK as sitk
import numpy as np
import pydicom
from datetime import datetime


def _calculate_age(birth_date_str, study_date_str):
    """Вычисляет возраст пациента на момент исследования."""
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y%m%d")
        study_date = datetime.strptime(study_date_str, "%Y%m%d")
        age = study_date.year - birth_date.year - ((study_date.month, study_date.day) < (birth_date.month, birth_date.day))
        return age
    except Exception:
        return "Unknown"


def _compute_slice_nonuniformity(dicom_series):
    z_positions = []

    for file in dicom_series:
        try:
            ds = pydicom.dcmread(file, stop_before_pixels=True)
            ipp = ds.get("ImagePositionPatient", None)
            if ipp:
                z_positions.append(ipp[2])  # Z-координата
        except Exception:
            continue

    if len(z_positions) < 2:
        return "Insufficient slices"

    z_positions = sorted(z_positions)
    diffs = np.diff(z_positions)

    max_diff = np.max(diffs)
    min_diff = np.min(diffs)
    nonuniformity = max_diff - min_diff

    return round(nonuniformity, 5)


def check_dcm_info(dicom_folder: str):

    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)

    if not dicom_series:
        return {"error": "No DICOM files found."}

    # Читаем метаданные через pydicom (первый файл)
    ds = pydicom.dcmread(dicom_series[0])
    birth_date = ds.get("PatientBirthDate", "")
    study_date = ds.get("StudyDate", ds.get("SeriesDate", ""))
    age = _calculate_age(birth_date, study_date)
    sex = ds.get("PatientSex", "Unknown")
    manufacturer = ds.get("Manufacturer", "Unknown")
    model = ds.get("ManufacturerModelName", "Unknown")
    slice_thickness = ds.get("SliceThickness", "Unknown")

    # Вычисляем неравномерность по оси Z
    z_nonuniformity = _compute_slice_nonuniformity(dicom_series)

    # Загружаем изображение и получаем геометрию через SimpleITK
    reader.SetFileNames(dicom_series)
    image = reader.Execute()

    img_size = image.GetSize()
    img_spacing = image.GetSpacing()
    img_origin = image.GetOrigin()
    img_direction = image.GetDirection()
    pixel_type = sitk.GetPixelIDValueAsString(image.GetPixelID())

    return {
        "PatientAge": age,
        "PatientSex": sex,
        "Manufacturer": manufacturer,
        "Model": model,
        "SliceThickness": slice_thickness,
        "ImageSize": img_size,
        "ImageSpacing": img_spacing,
        "ImageOrigin": img_origin,
        "ImageDirection": img_direction,
        "PixelType": pixel_type,
        "ZSpacingNonuniformity": z_nonuniformity
    }


def convert_dcm_to_nii(dicom_folder: str, nii_path: str, zip: bool = False):
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_series)
    image = reader.Execute()

    # Приведение к стандартной ориентации и direction
    identity_direction = (1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0)
    if image.GetDirection() != identity_direction:
        image = sitk.DICOMOrient(image, 'LPS')
        image.SetDirection(identity_direction)

    # Приведение к float32
    image = sitk.Cast(image, sitk.sitkInt16)

    # Приведение к новому spacing
    new_spacing = [0.4, 0.4, 0.4]
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resampled = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),  # origin не трогаем
        new_spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID()
    )

    out_path = nii_path + ".nii.gz" if zip else nii_path + ".nii"
    sitk.WriteImage(resampled, out_path)


def reader_dcm(dicom_folder: str):

    # Reading a series of DICOM files
    reader = sitk.ImageSeriesReader()

    # Getting a list of DICOM files in the specified folder
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)

    # Installing files in the reader
    reader.SetFileNames(dicom_series)

    # Reading images
    image = reader.Execute()

    img_size = image.GetSize()
    img_origin = image.GetOrigin()
    img_spacing = image.GetSpacing()
    img_direction = image.GetDirection()

    return img_size, img_origin, img_spacing, img_direction


def resample_nii(nii_original_path: str,
                 nii_resample_path: str,
                 size_or_pixel: str = None,
                 variable: list[float] = [1.0, 1.0, 1.0]):

    # Loading the original image
    image = sitk.ReadImage(nii_original_path)

    # Create a resampling filter
    resampler = sitk.ResampleImageFilter()

    # Set new voxel sizes (e.g. 1x1x1 mm)
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    if size_or_pixel == "size":
        # Вычисляем новое `spacing` (чтобы сохранить правильный масштаб)
        new_size = variable
        new_spacing = [
            (original_spacing[0] * original_size[0]) / new_size[0],
            (original_spacing[1] * original_size[1]) / new_size[1],
            (original_spacing[2] * original_size[2]) / new_size[2]
        ]
    else:
        new_spacing = variable
        # Calculate the new image size in voxels
        new_size = [
            int(original_size[0] * (original_spacing[0] / new_spacing[0])),
            int(original_size[1] * (original_spacing[1] / new_spacing[1])),
            int(original_size[2] * (original_spacing[2] / new_spacing[2]))
        ]

    # Set parameters resampling
    resampler.SetOutputSpacing(new_spacing)
    # resampler.SetSize(tuple(new_size))
    resampler.SetSize(np.array(new_size, dtype='int').tolist())
    resampler.SetOutputOrigin(image.GetOrigin())  # Initial origin point
    resampler.SetOutputDirection(image.GetDirection())  # Keep the same orientation

    # Interpolation: linear, Nearest-neighbor, B-Spline of order 3 interpolation
    # resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetInterpolator(sitk.sitkBSpline)
    # resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    # Set the value for pixels that will be outside the image (for example, 0)
    resampler.SetDefaultPixelValue(0)

    # Apply the resampling filter
    resampled_image = resampler.Execute(image)

    # Saving an image in NIfTI format
    sitk.WriteImage(resampled_image, nii_resample_path)


if __name__ == "__main__":
    convert_dcm_to_nii("C:/Users/Kamil/Aortic_valve/data/Homburg pathology DICOM/HOM_M19_H217_W96_YA_MJ",
                       "C:/Users/Kamil/Aortic_valve/data/Homburg pathology nii")
                    #"o_HOM_M19_H217_W96_YA.txt")