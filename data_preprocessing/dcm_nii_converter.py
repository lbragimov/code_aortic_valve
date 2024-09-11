from pathlib import Path

import SimpleITK as sitk
import numpy as np
import nibabel as nib

def convert_from_dicom_to_nifti(dicom_directory: str):
    # dir_name = Path(dicom_directory).parts[-1]
    # output_folder = Path(dicom_directory).parent
    # output_file = str(output_folder) + '\\' + dir_name + '.nii.gz'
    # series_reader = sitk.ImageSeriesReader()
    # series_dicom_names = series_reader.GetGDCMSeriesFileNames(dicom_directory)
    # series_reader.SetFileNames(series_dicom_names)
    # image3d = series_reader.Execute()

    directory_name = Path(dicom_directory).parts[-1]
    output_folder = Path(dicom_directory).parent

    if directory_name.startswith('RTG-LAT-preop'):
        # axes = [['L', 'R'], ['S', 'I'], ['A', 'P']]
        # # Получение всех возможных комбинаций
        # axis = list(itertools.product(*axes))
        # result_list = []
        # for pre_version in axis:
        #     for final_version in itertools.permutations(pre_version, 3):
        #         result_list.append((''.join(final_version)))
        result_list = ['PSL']
        for option in result_list:
            series_reader = sitk.ImageSeriesReader()
            series_dicom_names = series_reader.GetGDCMSeriesFileNames(dicom_directory)
            series_reader.SetFileNames(series_dicom_names)
            image = series_reader.Execute()

            orientation_filter = sitk.DICOMOrientImageFilter()
            orientation_filter.SetDesiredCoordinateOrientation(option)
            image = orientation_filter.Execute(image)

            image.SetOrigin((0, 0, 0))

            # Get original image spacing
            original_spacing = image.GetSpacing()

            # Define the new spacing (isotropic)
            new_spacing = [1, 1, 1]

            # Compute new image size
            original_size = np.array(image.GetSize(), dtype=np.int)
            new_size = original_size * (original_spacing / np.array(new_spacing))
            new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
            new_size = [int(s) for s in new_size]  # SimpleITK expects lists, not ndarrays

            # Set up the resampler
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetSize(new_size)
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetTransform(sitk.Transform())
            resampler.SetDefaultPixelValue(image.GetPixelIDValue())
            resampler.SetInterpolator(sitk.sitkBSpline)

            # Apply the resampling
            reoriented = resampler.Execute(image)

            # Write the transformed image back to a file
            output_file = str(output_folder) + '\\' + directory_name + '_' + option + '.nii.gz'
            # reoriented = reoriented[::-1, ::-1, ::-1]
            sitk.WriteImage(reoriented, output_file)

            image_nii_before = nib.load(output_file)
            array_before = image_nii_before.get_fdata()
            array = nib.orientations.flip_axis(array_before, axis=0)
            array = nib.orientations.flip_axis(array, axis=1)
            new_nifti = nib.Nifti1Image(array, image_nii_before.affine)
            nib.save(new_nifti, output_file)
    else:
        pass