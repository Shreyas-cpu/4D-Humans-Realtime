import os
import sys
import json
import urllib.request
import zipfile
import concurrent.futures

os.makedirs("wheels", exist_ok=True)
os.makedirs("pkgs", exist_ok=True)

def download_and_extract_pypi(package_name, version=None):
    try:
        api_url = f"https://pypi.org/pypi/{package_name}/json"
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        
        releases = data.get("releases", {})
        if version and version in releases:
            urls = releases[version]
        else:
            urls = data.get("urls", [])
        
        chosen = None
        for u in urls:
            fn = u["filename"]
            if fn.endswith(".whl"):
                if "cp310" in fn and ("manylinux" in fn or "linux" in fn):
                    chosen = u
                    break
                elif "py3-none-any" in fn or "py2.py3-none-any" in fn or "py3-none-manylinux" in fn:
                    chosen = u
                elif not chosen:
                    chosen = u
        
        if not chosen:
            print(f"[!] No suitable wheel for {package_name}")
            return False

        url = chosen["url"]
        fn = chosen["filename"]
        wheel_path = os.path.join("wheels", fn)
        
        # Download
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp, open(wheel_path, "wb") as f:
            while True:
                buf = resp.read(1024 * 1024)
                if not buf:
                    break
                f.write(buf)

        # Extract
        with zipfile.ZipFile(wheel_path, "r") as z:
            z.extractall("pkgs")
        print(f"[+] Installed {package_name} ({fn})")
        return True
    except Exception as e:
        print(f"[-] Failed {package_name}: {e}")
        return False

packages = [
    ("typing-extensions", "4.12.2"),
    ("filelock", "3.16.1"),
    ("sympy", "1.13.3"),
    ("mpmath", "1.3.0"),
    ("networkx", "3.3"),
    ("jinja2", "3.1.4"),
    ("MarkupSafe", "2.1.5"),
    ("fsspec", "2024.6.1"),
    ("pillow", "10.4.0"),
    ("numpy", "1.26.4"),
    ("smplx", "0.1.28"),
    ("pyrender", "0.1.45"),
    ("opencv-python", "4.10.0.84"),
    ("scikit-image", "0.24.0"),
    ("scipy", "1.13.1"),
    ("PyWavelets", "1.6.0"),
    ("imageio", "2.34.2"),
    ("tifffile", "2024.7.24"),
    ("lazy_loader", "0.4"),
    ("einops", "0.8.0"),
    ("timm", "1.0.7"),
    ("dill", "0.3.8"),
    ("pandas", "2.2.2"),
    ("pytz", "2024.1"),
    ("python-dateutil", "2.9.0.post0"),
    ("six", "1.16.0"),
    ("rich", "13.7.1"),
    ("markdown-it-py", "3.0.0"),
    ("mdurl", "0.1.2"),
    ("pygments", "2.18.0"),
    ("hydra-core", "1.3.2"),
    ("omegaconf", "2.3.0"),
    ("antlr4-python3-runtime", "4.9.3"),
    ("hydra-colorlog", "1.2.0"),
    ("colorama", "0.4.6"),
    ("pyrootutils", "1.0.4"),
    ("python-dotenv", "1.0.1"),
    ("webdataset", "0.2.86"),
    ("braceexpand", "0.1.7"),
    ("gdown", "5.2.0"),
    ("beautifulsoup4", "4.12.3"),
    ("soupsieve", "2.5"),
    ("requests", "2.32.3"),
    ("urllib3", "2.2.2"),
    ("certifi", "2024.7.4"),
    ("charset-normalizer", "3.3.2"),
    ("idna", "3.7"),
    ("tqdm", "4.66.4"),
    ("pytorch-lightning", "2.3.3"),
    ("torchmetrics", "1.4.0.post0"),
    ("lightning-utilities", "0.11.6"),
    ("gradio", "4.37.2"),
    ("gradio-client", "1.0.2"),
    ("fastapi", "0.111.0"),
    ("starlette", "0.37.2"),
    ("pydantic", "2.8.2"),
    ("pydantic-core", "2.20.1"),
    ("uvicorn", "0.30.1"),
    ("websockets", "12.0"),
    ("pyopengl", "3.1.7"),
    ("pyopengl-accelerate", "3.1.7"),
    ("trimesh", "4.4.1"),
    ("fvcore", "0.1.5.post20221221"),
    ("iopath", "0.1.10"),
    ("pytube", "15.0.0"),
    ("pyglet", "1.5.28"),
    ("chumpy", "0.70"),
    ("tabulate", "0.9.0"),
    ("termcolor", "2.4.0"),
    ("portalocker", "2.10.0"),
]

with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    futures = [executor.submit(download_and_extract_pypi, pkg, ver) for pkg, ver in packages]
    concurrent.futures.wait(futures)

print("Batch installation completed!")
