import os
import shutil
import copy
import SimpleITK as sitk
from tkinter import messagebox, Tk
from slicer_project_generator.scripts.utils import json_reader, json_save
import re

# from pyarrow import output_stream

template_point = {
    "id": "",
    "label": "",
    "description": "",
    "associatedNodeID": "",
    "position": [],
    "orientation": [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    "selected": True,
    "locked": True,
    "visibility": True,
    "positionStatus": "defined"
}

TEMPLATE_MAP = {
    "BasalRing.json":      ("BR - closed",     "BasalRing"),
    "CuspInsertion_L.json": ("LCI",            "CuspInsertion_L"),
    "CuspInsertion_N.json": ("NCI",            "CuspInsertion_N"),
    "CuspInsertion_R.json": ("RCI",            "CuspInsertion_R"),
    "GeometricHeight_L.json": ("LGH",          "GeometricHeight_L"),
    "GeometricHeight_N.json": ("NGH",          "GeometricHeight_N"),
    "GeometricHeight_R.json": ("RGH",          "GeometricHeight_R"),
    "Sinus_LN.json":        ("LNS - closed",   "Sinus_LN"),
    "Sinus_RL.json":        ("RLS - closed",   "Sinus_RL"),
    "Sinus_RN.json":        ("RNS - closed",   "Sinus_RN"),
}

def fill_template(case_data, template, key, label_prefix, assoc_node="vtkMRMLScalarVolumeNode1"):
    points_list = []
    for n, point in enumerate(case_data[key], start=1):
        temp_point = copy.deepcopy(template_point)
        temp_point["id"] = str(n)
        temp_point["label"] = f"{label_prefix}_{n}"
        temp_point["associatedNodeID"] = assoc_node
        temp_point["position"] = point
        points_list.append(temp_point)
    template["markups"][0]["controlPoints"] = points_list
    template["markups"][0]["lastUsedControlPointNumber"] = len(points_list)
    return template

def ref_points(case_data, template):
    allowed_keys = ["R", "L", "N", "RLC", "RNC", "LNC"]
    points_list = []
    n = 1
    for key, point in case_data.items():
        if not key in allowed_keys:
            continue
        temp_point = copy.deepcopy(template_point)
        temp_point["id"] = str(n)
        temp_point["label"] = key
        temp_point["position"] = point[0]
        points_list.append(temp_point)
        n += 1
    template["markups"][0]["controlPoints"] = points_list
    template["markups"][0]["lastUsedControlPointNumber"] = len(points_list)
    return template


def convert_nii_to_nrrd(input_path, output_path, check=True, is_mask=False):
    image = sitk.ReadImage(input_path)

    if is_mask:
        image = sitk.Cast(image, sitk.sitkUInt8)

    sitk.WriteImage(image, output_path, useCompression=True)

    attention = ""
    if check:
        # Read back the saved NRRD
        converted = sitk.ReadImage(output_path)

        # Checking for property matches
        checks = {
            "Size": image.GetSize() == converted.GetSize(),
            "Spacing": all(abs(a - b) < 1e-6 for a, b in zip(image.GetSpacing(), converted.GetSpacing())),
            "Origin": all(abs(a - b) < 1e-6 for a, b in zip(image.GetOrigin(), converted.GetOrigin())),
            "Direction": all(abs(a - b) < 1e-6 for a, b in zip(image.GetDirection(), converted.GetDirection())),
        }

        for key, result in checks.items():
            if not result:
                attention = "ATTENTION: there are discrepancies between NIfTI and NRRD!"
                break
    return attention

def mrml_generator(file_path, nii_img_path, output_file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # читаем данные из nii
    image = sitk.ReadImage(nii_img_path)
    origin = image.GetOrigin()
    spacing = image.GetSpacing()

    new_origin = f'{origin[0]} {origin[1]} {origin[2]}'
    new_spacing = f'{spacing[0]} {spacing[1]} {spacing[2]}'

    new_lines = []
    for line in lines:
        if "<Volume " in line and "vtkMRMLScalarVolumeNode1" in line:
            # заменяем spacing
            line = re.sub(r'spacing="[^"]+"', f'spacing="{new_spacing}"', line)
            # заменяем origin
            line = re.sub(r'origin="[^"]+"', f'origin="{new_origin}"', line)
        new_lines.append(line)

    # сохраняем изменения обратно
    with open(output_file_path, "w") as f:
        f.writelines(new_lines)


class ProjectGenerator:
    def __init__(self, case_name, output_folder, original_img_folder, original_aorta_mask_folder, case_data,
                 base_path="cases"):
        self.case_name = case_name
        self.output_folder = output_folder
        self.original_img_folder = original_img_folder
        self.original_aorta_mask_folder = original_aorta_mask_folder
        self.case_data = case_data
        # self.base_path = base_path
        self.templates_folder = "templates"
        self.attentions = []

    def check_and_prepare_folder(self, case_folder_path):
        """Проверяет наличие папки и при необходимости очищает её."""
        root = Tk()
        root.withdraw()

        if os.path.exists(case_folder_path):
            answer = messagebox.askyesno(
                "Folder already exists",
                f"The folder '{case_folder_path}' already exists.\n"
                f"Do you want to overwrite it with new files?"
            )
            if not answer:
                messagebox.showinfo("Canceled", "Project generation canceled.")
                root.destroy()
                return False

            shutil.rmtree(case_folder_path)

        os.makedirs(case_folder_path, exist_ok=True)
        root.destroy()
        return True

    def copy_templates(self, case_folder_path):
        """Копирует все файлы из папки templates в папку проекта."""
        attention = convert_nii_to_nrrd(
            os.path.join(self.original_img_folder, self.case_name + ".nii.gz"),
            os.path.join(case_folder_path, "CT_img.nrrd")
        )
        if attention:
            self.attentions.append(f"CT_img.nrrd: {attention}")
        attention = convert_nii_to_nrrd(
            os.path.join(self.original_aorta_mask_folder, self.case_name + ".nii.gz"),
            os.path.join(case_folder_path, "SegMask.seg.nrrd"), is_mask=True
        )
        if attention:
            self.attentions.append(f"SegMask.seg.nrrd: {attention}")

        for filename in os.listdir(self.templates_folder):
            if filename.endswith(".mrml"):
                mrml_generator(os.path.join(self.templates_folder, filename),
                               os.path.join(self.original_img_folder, self.case_name + ".nii.gz"),
                               os.path.join(case_folder_path, filename))
                continue
            elif filename.endswith(".json"):
                template = json_reader(os.path.join(self.templates_folder, filename))
                if filename in TEMPLATE_MAP:
                    key, label_prefix = TEMPLATE_MAP[filename]
                    new_json = fill_template(self.case_data, template, key, label_prefix)
                elif filename == 'RefPoints.json':
                    new_json = ref_points(self.case_data, template)
                else:
                    continue
                json_save(new_json, os.path.join(case_folder_path, filename))

    def generate(self):
        case_folder_path = os.path.join(self.output_folder, self.case_name)
        print("hi")

        if not self.check_and_prepare_folder(case_folder_path):
            return None

        self.copy_templates(case_folder_path)

        # сообщение пользователю
        root = Tk()
        root.withdraw()
        msg = f"Project for case '{self.case_name}' created in:\n{case_folder_path}"
        if self.attentions:
            msg += "\n\n" + "\n".join(self.attentions)

        messagebox.showinfo("Success", msg)
        root.destroy()

        return case_folder_path

        # output_file = f"{self.case_name}_project.xml"

#         # пока простейший xml
#         xml_content = f"""<?xml version="1.0"?>
# <Project>
#     <Case>{self.case_name}</Case>
#     <Path>{case_path}</Path>
# </Project>
# """
#         with open(output_file, "w", encoding="utf-8") as f:
#             f.write(xml_content)
#
#         return output_file