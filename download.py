"""
Downloader
"""
#libraries
import requests

#url
url = "http://datashare.is.ed.ac.uk/download/DS_10283_3443.zip"

#cehck if file is created

#if not create it

#if yes don't do anything

#download the file and set the destination as dataset inside this repo
r = requests.get(url, Stream = True)
with open("dataset/VCTK.zip", "wb") as dataset:
    for chunk in r.iter_content(chunk_size=1024)
        if chunk:
            dataset.write(chunk)