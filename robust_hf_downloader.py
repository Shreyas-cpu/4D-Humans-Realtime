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
req = urllib.request.Request(initial_url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'})
with urllib.request.urlopen(req, timeout=20) as resp:
    final_url = resp.url
    total_size = int(resp.headers.get('Content-Length', 0))

print(f"Resolved URL: {total_size / (1024*1024):.1f} MB ({total_size} bytes)")

num_threads = 6
chunk_size = total_size // num_threads

def download_part(idx, start_byte, end_byte, filename):
    expected = end_byte - start_byte + 1
    current = os.path.getsize(filename) if os.path.exists(filename) else 0
    while current < expected:
        r_start = start_byte + current
        req = urllib.request.Request(final_url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)',
            'Range': f'bytes={r_start}-{end_byte}'
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(filename, 'ab') as f:
                while True:
                    buf = resp.read(1024 * 1024)
                    if not buf:
                        break
                    f.write(buf)
                    current += len(buf)
        except Exception as e:
            time.sleep(1)
            current = os.path.getsize(filename) if os.path.exists(filename) else 0
    assert os.path.getsize(filename) == expected, f"Part {idx} mismatch"
    print(f"Part {idx} ({expected/(1024*1024):.1f} MB) completed!")

part_files = []
futures = []
t0 = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    for i in range(num_threads):
        start = i * chunk_size
        end = (start + chunk_size - 1) if (i < num_threads - 1) else (total_size - 1)
        pfile = f"{target_ckpt}.part{i}"
        part_files.append(pfile)
        futures.append(executor.submit(download_part, i, start, end, pfile))
        
    concurrent.futures.wait(futures)
    for f in futures:
        f.result()

print(f"All parts finished in {time.time()-t0:.1f}s. Assembling...")
with open(target_ckpt, "wb") as outfile:
    for pf in part_files:
        with open(pf, "rb") as infile:
            while True:
                b = infile.read(1024 * 1024 * 8)
                if not b:
                    break
                outfile.write(b)
        try:
            os.remove(pf)
        except:
            pass

print(f"Assembled {target_ckpt} ({os.path.getsize(target_ckpt)} bytes). Verifying...")
with zipfile.ZipFile(target_ckpt, "r") as z:
    bad = z.testzip()
    if bad:
        print("Corrupted:", bad)
    else:
        print("SUCCESS! Checkpoint is 100% valid and verified!")
