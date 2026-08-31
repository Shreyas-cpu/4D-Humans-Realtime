import os
import sys
import tarfile
from multithread_downloader import download_file_multithread

cache_dir = os.path.expanduser("~/.cache/4DHumans")
os.makedirs(cache_dir, exist_ok=True)
tar_path = os.path.join(cache_dir, "hmr2_data.tar.gz")

url = "https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz"
print(f"Downloading {url} to {tar_path}...")
download_file_multithread(url, tar_path, num_threads=12)

print(f"Extracting {tar_path} to {cache_dir}...")
with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(cache_dir)

print(f"Also ensuring data/ directory in workspace is populated...")
os.makedirs("data", exist_ok=True)
if os.path.exists(os.path.join(cache_dir, "data")):
    import shutil
    for item in os.listdir(os.path.join(cache_dir, "data")):
        s = os.path.join(cache_dir, "data", item)
        d = os.path.join("data", item)
        if os.path.isdir(s) and not os.path.exists(d):
            shutil.copytree(s, d)
        elif not os.path.exists(d):
            shutil.copy2(s, d)

print("HMR2 models & data extraction complete!")
