from data_preprocessing.text_worker import add_info_logging
import os
from pathlib import Path
import json
import numpy as np
from pycpd import AffineRegistration



def controller(data_path):
    ds_folder_name = "Dataset499_AortaLandmarks"
    add_info_logging("Finish", "work_logger")


if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    controller(data_path)