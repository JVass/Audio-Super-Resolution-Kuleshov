"""
Unzipper and folder management
"""
import shutil
import os

def check_dataset_folder():
    return os.path.isdir("./dataset/extracted_data/")


def instantiate_file():
    if not(check_dataset_folder()):
        os.mkdir("./dataset/extracted_data/")
    else:
        print("Dataset folder is already initialized, initiating downloading procedure...")

instantiate_file()

print("Start the extraction procedure. Grab another coffee...")
shutil.unpack_archive("./dataset/VCTK.zip", "./dataset")