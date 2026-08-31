import os
import sys
import time
import urllib.request
import concurrent.futures
import zipfile

target_dir = os.path.expanduser("~/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints")
os.makedirs(target_dir, exist_ok=True)
target_ckpt = os.path.join(target_dir, "epoch=35-step=1000000.ckpt")

initial_url = "https://huggingface.co/lamianlbe/4D-Humans/resolve/main/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"

print(f"Resolving redirect for {initial_url} ...")
req = urllib.request.Request(initial_url, headers={'User-Agent': 'Mozilla/5.0'}, method='HEAD')
with urllib.request.urlopen(req, timeout=20) as resp:
    final_url = resp.url
    total_size = int(resp.headers.get('Content-Length', 0))

print(f"Direct CDN URL obtained! Total size: {total_size / (1024*1024):.2f} MB ({total_size} bytes)")

def download_chunk(part_idx, start_byte, end_byte, part_file):
    expected_size = end_byte - start_byte + 1
    current_size = os.path.getsize(part_file) if os.path.exists(part_file) else 0
    
    while current_size < expected_size:
        req_start = start_byte + current_size
        req = urllib.request.Request(final_url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Range': f'bytes={req_start}-{end_byte}'
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(part_file, 'ab') as f:
                while True:
                    buf = resp.read(1024 * 1024 * 4) # 4MB
                    if not buf:
                        break
                    f.write(buf)
                    current_size += len(buf)
        except Exception as e:
            time.sleep(0.5)
            current_size = os.path.getsize(part_file) if os.path.exists(part_file) else 0

    assert os.path.getsize(part_file) == expected_size

num_threads = 16
chunk_size = total_size // num_threads
part_files = []
futures = []

print(f"Downloading checkpoint across {num_threads} parallel threads from AWS CDN...")
t0 = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    for i in range(num_threads):
        start = i * chunk_size
        end = (start + chunk_size - 1) if (i < num_threads - 1) else (total_size - 1)
        part_file = f"{target_ckpt}.part{i}"
        part_files.append(part_file)
        futures.append(executor.submit(download_chunk, i, start, end, part_file))
        
    concurrent.futures.wait(futures)
    for f in futures:
        f.result()

print(f"All {num_threads} parts downloaded in {time.time()-t0:.1f}s! Assembling {target_ckpt}...")
with open(target_ckpt, "wb") as outfile:
    for part_file in part_files:
        with open(part_file, "rb") as infile:
            while True:
                buf = infile.read(1024 * 1024 * 16)
                if not buf:
                    break
                outfile.write(buf)
        try:
            os.remove(part_file)
        except:
            pass

print(f"Assembled {target_ckpt}: {os.path.getsize(target_ckpt)} bytes.")
print("Testing checkpoint zip integrity...")
with zipfile.ZipFile(target_ckpt, "r") as z:
    bad = z.testzip()
    if bad:
        print(f"Error: Corrupted file inside zip: {bad}")
    else:
        print("PERFECT! Checkpoint zip integrity 100% verified!")
