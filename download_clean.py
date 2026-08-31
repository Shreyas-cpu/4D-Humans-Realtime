import urllib.request
import os
import time

url = "https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz"
out_path = os.path.expanduser("~/.cache/4DHumans/hmr2_clean.tar")

print(f"Downloading {url} to {out_path}...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=60) as resp, open(out_path, "wb") as f:
    total_size = int(resp.headers.get('Content-Length', 0))
    print(f"Total size to download: {total_size / (1024*1024):.1f} MB")
    downloaded = 0
    t0 = time.time()
    last_print = t0
    
    while True:
        chunk = resp.read(1024 * 1024 * 4) # 4MB
        if not chunk:
            break
        f.write(chunk)
        downloaded += len(chunk)
        now = time.time()
        if now - last_print > 5:
            speed = downloaded / (1024*1024 * (now - t0))
            percent = (downloaded / total_size) * 100
            print(f"{downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB ({percent:.1f}%) @ {speed:.2f} MB/s")
            last_print = now

print(f"Download complete! Final size: {os.path.getsize(out_path)} bytes")
