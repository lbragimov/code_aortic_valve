# scripts/generator.py
import os

class ProjectGenerator:
    def __init__(self, case_name, base_path="cases"):
        self.case_name = case_name
        self.base_path = base_path

    def generate(self):
        case_path = os.path.join(self.base_path, self.case_name)
        output_file = f"{self.case_name}_project.xml"

        # пока простейший xml
        xml_content = f"""<?xml version="1.0"?>
<Project>
    <Case>{self.case_name}</Case>
    <Path>{case_path}</Path>
</Project>
"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(xml_content)

        return output_file