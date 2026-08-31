import os
import sys
import json
import urllib.request
import zipfile
import concurrent.futures

os.makedirs("wheels", exist_ok=True)
os.makedirs("pkgs", exist_ok=True)

def download_and_extract_pypi(package_name):
    try:
        api_url = f"https://pypi.org/pypi/{package_name}/json"
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        
        urls = data.get("urls", [])
        chosen = None
        for u in urls:
            fn = u["filename"]
            if fn.endswith(".whl") and ("manylinux" in fn or "linux" in fn or "any" in fn):
                chosen = u
                break
        
        if not chosen:
            print(f"[!] No suitable wheel for {package_name}")
            return False

        url = chosen["url"]
        fn = chosen["filename"]
        wheel_path = os.path.join("wheels", fn)
        
        print(f"Downloading {fn}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as resp, open(wheel_path, "wb") as f:
            while True:
                buf = resp.read(1024 * 1024 * 2)
                if not buf:
                    break
                f.write(buf)

        with zipfile.ZipFile(wheel_path, "r") as z:
            z.extractall("pkgs")
        print(f"[+] Successfully installed {package_name} ({fn})")
        return True
    except Exception as e:
        print(f"[-] Failed {package_name}: {e}")
        return False

nvidia_pkgs = [
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvtx-cu12",
    "nvidia-nvjitlink-cu12",
]

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(download_and_extract_pypi, pkg) for pkg in nvidia_pkgs]
    concurrent.futures.wait(futures)

print("CUDA libraries installation completed!")
