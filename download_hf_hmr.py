import os
import tarfile
from multithread_downloader import download_file_multithread

cache_dir = os.path.expanduser("~/.cache/4DHumans")
os.makedirs(cache_dir, exist_ok=True)
tar_path = os.path.join(cache_dir, "hmr2_hf.tar.gz")

url = "https://huggingface.co/camenduru/4D-Humans/resolve/main/hmr2_data.tar.gz"
print(f"Downloading {url} to {tar_path}...")
download_file_multithread(url, tar_path, num_threads=12)

print(f"Extracting {tar_path} to {cache_dir}...")
with tarfile.open(tar_path, "r:*") as tar:
    tar.extractall(cache_dir)

print("Extraction complete! Verifying files...")
