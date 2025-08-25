import os

def get_available_cases(cases_dir="cases"):
    if not os.path.exists(cases_dir):
        return []
    return [d for d in os.listdir(cases_dir) if os.path.isdir(os.path.join(cases_dir, d))]