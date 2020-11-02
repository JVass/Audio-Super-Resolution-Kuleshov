"""
Downloader
"""
#libraries
import requests
import os

def check_dataset_folder():
    return os.path.isdir("./dataset")

def instantiate_file():
    if not(check_dataset_folder()):
        os.mkdir("./dataset")
    else:
        print("Dataset folder is already installed, initiating downloading procedure...")

#url
url = "http://datashare.is.ed.ac.uk/download/DS_10283_3443.zip"

instantiate_file()

#download the file and set the destination as dataset inside this repo
r = requests.get(url, Stream = True)

with open("dataset/VCTK.zip", "wb") as dataset:
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:
            dataset.write(chunk)