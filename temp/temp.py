import os
from pathlib import Path
from datetime import datetime
from data_preprocessing.text_worker import add_info_logging
from data_preprocessing.text_worker import json_reader
import SimpleITK as sitk
import pydicom


def calculate_age(birth_date_str, study_date_str):
    """Вычисляет возраст пациента на момент исследования."""
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y%m%d")
        study_date = datetime.strptime(study_date_str, "%Y%m%d")
        age = study_date.year - birth_date.year - ((study_date.month, study_date.day) < (birth_date.month, birth_date.day))
        return age
    except Exception:
        return "Unknown"


def check_dcm_info(dicom_folder: str):

    # Reading a series of DICOM files
    reader = sitk.ImageSeriesReader()

    # Getting a list of DICOM files in the specified folder
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)

    # Читаем первый файл серии для извлечения метаданных
    first_file = dicom_series[0]
    ds = pydicom.dcmread(first_file)

    # Извлекаем данные
    birth_date = ds.get("PatientBirthDate", "")
    study_date = ds.get("StudyDate", ds.get("SeriesDate", ""))
    age = calculate_age(birth_date, study_date)

    # Извлекаем необходимые поля
    sex = ds.get("PatientSex", "Unknown")
    slice_thickness = ds.get("SliceThickness", "Unknown")
    manufacturer = ds.get("Manufacturer", "Unknown")
    model = ds.get("ManufacturerModelName", "Unknown"),

    return {
        "PatientAge": age,
        "PatientSex": sex,
        "SliceThickness": slice_thickness,
        "Manufacturer": manufacturer,
        "Model": model
    }


def controller(data_path):
    result_path = os.path.join(data_path, "result")
    dicom_path = os.path.join(data_path, "dicom")
    script_dir = Path(__file__).resolve().parent
    data_structure_path = os.path.join(script_dir, "dir_structure.json")
    dir_structure = json_reader(data_structure_path)
    add_info_logging("Start", "work_logger")


    summary_info = {
        "PatientAge": set(),
        "PatientSex": set(),
        "SliceThickness": set(),
        "Manufacturer": set(),
        "Model": set()
    }
    for sub_dir in list(dir_structure['dicom']):
        for case in os.listdir(os.path.join(dicom_path, sub_dir)):
            dcm_case_path = os.path.join(dicom_path, sub_dir, case)
            info = check_dcm_info(dcm_case_path)
            for key in summary_info:
                summary_info[key].add(info[key])

    add_info_logging("Finish", "work_logger")


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller(data_path)