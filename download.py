"""
Downloader
"""
#libraries
import requests
import os
import ssl

chunk_size = 1024

def check_dataset_folder():
    return os.path.isdir("./dataset")

def instantiate_file():
    if not(check_dataset_folder()):
        os.mkdir("./dataset")
    else:
        print("Dataset folder is already initialized, initiating downloading procedure...")

#url
url = "http://datashare.is.ed.ac.uk/download/DS_10283_3443.zip"

instantiate_file()

context = ssl.SSLContext()

#download the file and set the destination as dataset inside this repo
r = requests.get(url, stream = True, verify = False)

total_size = 8074538516.48 #bits
total_chunks_num = total_size // chunk_size

flag = -1

with open("dataset/VCTK.zip", "wb") as dataset:
    for chunk_counter, chunk in enumerate(r.iter_content(chunk_size=chunk_size)):
        if chunk:
            dataset.write(chunk)

        file_percent_downloaded = chunk_counter / total_chunks_num * 100

        if  (int(file_percent_downloaded) % 5 == 0) and (flag < int(file_percent_downloaded)):  
            if flag == -1:
                print("Download started. Grab a coffee. You're gonna need it...")
                flag = 0
            else:
                print("Finished {}%".format(int(file_percent_downloaded)))

                flag = file_percent_downloaded

    print("Finished downloading. Next step: unzipping.")
        