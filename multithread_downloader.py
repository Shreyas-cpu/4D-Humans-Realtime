import os
import sys
import json
import urllib.request
import concurrent.futures
import zipfile

os.makedirs("wheels", exist_ok=True)
os.makedirs("pkgs", exist_ok=True)

def download_chunk(url, start_byte, end_byte, part_filename):
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0',
        'Range': f'bytes={start_byte}-{end_byte}'
    })
    with urllib.request.urlopen(req, timeout=30) as resp, open(part_filename, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024 * 2) # 2MB
            if not chunk:
                break
            f.write(chunk)

def download_file_multithread(url, out_path, num_threads=8):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(f"File already exists: {out_path}")
        return
    
    print(f"Probing {url} ...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}, method='HEAD')
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            total_size = int(resp.headers.get('Content-Length', 0))
            accept_ranges = 'bytes' in resp.headers.get('Accept-Ranges', '')
    except Exception as e:
        print(f"HEAD request failed: {e}, falling back to single stream GET...")
        total_size = 0
        accept_ranges = False

    if total_size <= 0 or not accept_ranges or total_size < 10 * 1024 * 1024:
        # Small file or no ranges: standard download
        print(f"Downloading {os.path.basename(out_path)} directly ({total_size / (1024*1024):.1f} MB)...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp, open(out_path, "wb") as f:
            while True:
                buf = resp.read(1024 * 1024 * 2)
                if not buf:
                    break
                f.write(buf)
        print(f"Downloaded {os.path.basename(out_path)}")
        return

    print(f"Downloading {os.path.basename(out_path)}: {total_size / (1024*1024):.1f} MB across {num_threads} threads...")
    chunk_size = total_size // num_threads
    futures = []
    part_files = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start = i * chunk_size
            end = (start + chunk_size - 1) if (i < num_threads - 1) else (total_size - 1)
            part_file = f"{out_path}.part{i}"
            part_files.append(part_file)
            futures.append(executor.submit(download_chunk, url, start, end, part_file))
        
        concurrent.futures.wait(futures)
        for f in futures:
            f.result() # raise exceptions if any

    # Combine parts
    print(f"Merging parts for {os.path.basename(out_path)}...")
    with open(out_path, "wb") as outfile:
        for part_file in part_files:
            with open(part_file, "rb") as infile:
                while True:
                    b = infile.read(1024 * 1024 * 4)
                    if not b:
                        break
                    outfile.write(b)
            try:
                os.remove(part_file)
            except:
                pass
    print(f"Successfully downloaded {os.path.basename(out_path)} ({os.path.getsize(out_path)/(1024*1024):.1f} MB)!")

def extract_wheel(wheel_path):
    print(f"Extracting {os.path.basename(wheel_path)} into pkgs/...")
    with zipfile.ZipFile(wheel_path, "r") as z:
        z.extractall("pkgs")
    print(f"Extracted {os.path.basename(wheel_path)}!")

if __name__ == "__main__":
    url = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(url).split("?")[0]
    out_file = os.path.join("wheels", name)
    download_file_multithread(url, out_file)
    if out_file.endswith(".whl"):
        extract_wheel(out_file)
