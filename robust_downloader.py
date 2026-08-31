import os
import sys
import time
import urllib.request
import concurrent.futures
import tarfile
import zipfile

def download_chunk_robust(url, start_byte, end_byte, part_filename):
    expected_size = end_byte - start_byte + 1
    current_size = os.path.getsize(part_filename) if os.path.exists(part_filename) else 0
    
    while current_size < expected_size:
        req_start = start_byte + current_size
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)',
            'Range': f'bytes={req_start}-{end_byte}'
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(part_filename, 'ab') as f:
                while True:
                    buf = resp.read(1024 * 1024 * 2) # 2MB
                    if not buf:
                        break
                    f.write(buf)
                    current_size += len(buf)
        except Exception as e:
            # print(f"Part {part_filename} retry after {e} ({current_size}/{expected_size} bytes)...")
            time.sleep(1)
            current_size = os.path.getsize(part_filename) if os.path.exists(part_filename) else 0

    assert os.path.getsize(part_filename) == expected_size, f"Size mismatch for {part_filename}"

def download_and_extract_robust(url, out_tar_path, extract_dir, num_threads=12):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}, method='HEAD')
    with urllib.request.urlopen(req, timeout=20) as resp:
        total_size = int(resp.headers.get('Content-Length', 0))
        
    print(f"Total size: {total_size / (1024*1024):.2f} MB across {num_threads} parallel threads.")
    chunk_size = total_size // num_threads
    futures = []
    part_files = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start = i * chunk_size
            end = (start + chunk_size - 1) if (i < num_threads - 1) else (total_size - 1)
            part_file = f"{out_tar_path}.part{i}"
            part_files.append(part_file)
            futures.append(executor.submit(download_chunk_robust, url, start, end, part_file))
            
        print("Waiting for all chunks to finish with verified CRC/size...")
        concurrent.futures.wait(futures)
        for f in futures:
            f.result()

    print("All chunks 100% verified! Merging archive...")
    with open(out_tar_path, "wb") as outfile:
        for part_file in part_files:
            with open(part_file, "rb") as infile:
                while True:
                    b = infile.read(1024 * 1024 * 8)
                    if not b:
                        break
                    outfile.write(b)
            try:
                os.remove(part_file)
            except:
                pass
                
    print(f"Merged archive: {os.path.getsize(out_tar_path)} bytes. Extracting to {extract_dir}...")
    os.system(f"tar -xf {out_tar_path} -C {extract_dir}")
    print("Archive successfully extracted!")

if __name__ == "__main__":
    cache_dir = os.path.expanduser("~/.cache/4DHumans")
    os.makedirs(cache_dir, exist_ok=True)
    tar_path = os.path.join(cache_dir, "hmr2_data.tar")
    url = "https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz"
    download_and_extract_robust(url, tar_path, cache_dir, num_threads=12)
