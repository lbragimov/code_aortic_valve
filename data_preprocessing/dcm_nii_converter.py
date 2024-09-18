from pathlib import Path

import SimpleITK as sitk
import numpy as np


def convert_dcm_to_nii(dicom_folder: str, nii_folder: str, zip: bool = False):

    if zip:
        output_nii_file = nii_folder + ".nii.gz"
    else:
        output_nii_file = nii_folder + ".nii"

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

    # Saving an image in NIfTI format
    sitk.WriteImage(image, output_nii_file)

    return img_size, img_origin, img_spacing, img_direction


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
                 new_spacing: list[float] = [1.0, 1.0, 1.0]):

    # Loading the original image
    image = sitk.ReadImage(nii_original_path)

    # Create a resampling filter
    resampler = sitk.ResampleImageFilter()

    # Set new voxel sizes (e.g. 1x1x1 mm)
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # Calculate the new image size in voxels
    new_size = [
        int(original_size[0] * (original_spacing[0] / new_spacing[0])),
        int(original_size[1] * (original_spacing[1] / new_spacing[1])),
        int(original_size[2] * (original_spacing[2] / new_spacing[2]))
    ]

    # Set parameters resampling
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
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