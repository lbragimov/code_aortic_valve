from scripts.gui import CaseSelector
from scripts.generator import ProjectGenerator
from scripts.utils import get_available_cases

def main():
    # get a list of cases
    cases = get_available_cases("cases")

    # show the selection window
    selector = CaseSelector(cases)
    selected_case = selector.run()

    if not selected_case:
        print("Case not selected, exit...")
        return

    # generate the project
    generator = ProjectGenerator(selected_case)
    project_file = generator.generate()

    print(f"Project created: {project_file}")

if __name__ == "__main__":
    main()