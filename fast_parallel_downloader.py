import os
import sys
import time
import requests
import concurrent.futures
import zipfile

target_dir = os.path.expanduser("~/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints")
os.makedirs(target_dir, exist_ok=True)
target_ckpt = os.path.join(target_dir, "epoch=35-step=1000000.ckpt")
url = "https://huggingface.co/lamianlbe/4D-Humans/resolve/main/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"

total_size = 2709521501
num_threads = 12
chunk_size = total_size // num_threads

print(f"Downloading {total_size / (1024*1024):.1f} MB checkpoint across {num_threads} streams...")

def download_part(idx, start_byte, end_byte, filename):
    expected = end_byte - start_byte + 1
    current = os.path.getsize(filename) if os.path.exists(filename) else 0
    s = requests.Session()
    
    while current < expected:
        r_start = start_byte + current
        headers = {
            'Range': f'bytes={r_start}-{end_byte}',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'
        }
        try:
            with s.get(url, headers=headers, stream=True, allow_redirects=True, timeout=30) as resp:
                if resp.status_code in [200, 206]:
                    with open(filename, 'ab') as f:
                        for chunk in resp.iter_content(chunk_size=1024*512):
                            if chunk:
                                f.write(chunk)
                                current += len(chunk)
        except Exception as e:
            time.sleep(1)
            current = os.path.getsize(filename) if os.path.exists(filename) else 0

    assert os.path.getsize(filename) == expected, f"Part {idx} mismatch ({os.path.getsize(filename)} vs {expected})"
    print(f"[+] Part {idx:02d} ({expected/(1024*1024):.1f} MB) 100% completed!")

part_files = []
futures = []
t0 = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    for i in range(num_threads):
        start = i * chunk_size
        end = (start + chunk_size - 1) if (i < num_threads - 1) else (total_size - 1)
        pfile = f"{target_ckpt}.chunk{i:02d}"
        part_files.append(pfile)
        futures.append(executor.submit(download_part, i, start, end, pfile))
        
    concurrent.futures.wait(futures)
    for f in futures:
        f.result()

print(f"All {num_threads} streams finished in {time.time()-t0:.1f}s! Assembling final checkpoint...")
with open(target_ckpt, "wb") as outfile:
    for pf in part_files:
        with open(pf, "rb") as infile:
            while True:
                b = infile.read(1024 * 1024 * 16)
                if not b:
                    break
                outfile.write(b)
        try:
            os.remove(pf)
        except:
            pass

print(f"Assembled {target_ckpt} ({os.path.getsize(target_ckpt)} bytes). Verifying zip...")
with zipfile.ZipFile(target_ckpt, "r") as z:
    bad = z.testzip()
    if bad:
        print("Corrupted:", bad)
    else:
        print("SUCCESS! Checkpoint is 100% valid and verified!")
