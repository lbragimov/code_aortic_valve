import os
from scripts.gui import CaseSelector
from scripts.generator import ProjectGenerator
from scripts.utils import get_available_cases, json_reader

def main(data_path):
    dict_all_case_path = os.path.join(data_path, "dict_all_case.json")
    output_folder = os.path.join(data_path, "result", "cases_visualization")
    original_img_folder = os.path.join(data_path, "image_nii_crop")
    original_aorta_mask_folder = os.path.join(data_path, "mask_aorta_segment")
    dict_all_case = json_reader(dict_all_case_path)
    # get a list of cases
    cases = list(dict_all_case.keys())

    # show the selection window
    selector = CaseSelector(cases)
    selected_case = selector.run()

    if not selected_case:
        print("Case not selected, exit...")
        return

    # generate the project
    generator = ProjectGenerator(case_name=selected_case,
                                 output_folder=output_folder,
                                 original_img_folder=original_img_folder,
                                 original_aorta_mask_folder=original_aorta_mask_folder,
                                 case_data=dict_all_case[selected_case])
    project_file = generator.generate()

    print(f"Project created: {project_file}")

if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    main(data_path)