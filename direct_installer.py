import os
import sys
import json
import urllib.request
import zipfile

os.makedirs("wheels", exist_ok=True)
os.makedirs("pkgs", exist_ok=True)

def download_and_extract(url, name=None):
    if not name:
        name = url.split("/")[-1].split("?")[0]
    wheel_path = os.path.join("wheels", name)
    
    if not os.path.exists(wheel_path) or os.path.getsize(wheel_path) == 0:
        print(f"Downloading {name} from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(wheel_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(1024 * 1024 * 4) # 4MB chunk
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int(downloaded * 100 / total)
                    print(f"\r  {downloaded / (1024*1024):.1f}MB / {total / (1024*1024):.1f}MB ({pct}%)", end="", flush=True)
            print()
    else:
        print(f"Already downloaded: {name}")

    print(f"Extracting {name} into pkgs/...")
    with zipfile.ZipFile(wheel_path, "r") as z:
        z.extractall("pkgs")
    print(f"Successfully extracted {name}!\n")

def download_pypi(package_name, version=None):
    api_url = f"https://pypi.org/pypi/{package_name}/json"
    req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    
    releases = data.get("releases", {})
    if version and version in releases:
        urls = releases[version]
    else:
        urls = data.get("urls", [])
    
    # Pick suitable wheel for cp310 manylinux or pure python
    chosen = None
    for u in urls:
        fn = u["filename"]
        if fn.endswith(".whl"):
            if "cp310" in fn and ("manylinux" in fn or "linux" in fn):
                chosen = u
                break
            elif "py3-none-any" in fn or "py2.py3-none-any" in fn or "py3-none-manylinux" in fn:
                chosen = u
                # Keep looking for cp310 if exists
            elif not chosen:
                chosen = u
    
    if not chosen:
        print(f"Could not find binary wheel for {package_name}")
        return
    
    download_and_extract(chosen["url"], chosen["filename"])

if __name__ == "__main__":
    # 1. PyTorch & CUDA wheels
    torch_wheels = [
        "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-linux_x86_64.whl",
        "https://download.pytorch.org/whl/cu121/triton-3.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "https://download.pytorch.org/whl/cu121/nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl",
        "https://download.pytorch.org/whl/cu121/nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl",
        "https://download.pytorch.org/whl/cu121/nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl",
        "https://download.pytorch.org/whl/cu121/nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl",
        "https://download.pytorch.org/whl/cu121/nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl",
        "https://download.pytorch.org/whl/cu121/nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl",
        "https://download.pytorch.org/whl/cu121/nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl",
        "https://download.pytorch.org/whl/cu121/nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl",
        "https://download.pytorch.org/whl/cu121/nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl",
        "https://download.pytorch.org/whl/cu121/nvidia_nccl_cu12-2.21.5-py3-none-manylinux2014_x86_64.whl",
        "https://download.pytorch.org/whl/cu121/nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl",
        "https://download.pytorch.org/whl/cu121/nvidia_nvjitlink_cu12-12.1.105-py3-none-manylinux1_x86_64.whl",
    ]
    for tw in torch_wheels:
        download_and_extract(tw)

    # 2. PyPI packages
    pypi_pkgs = [
        ("numpy", "1.26.4"),
        ("filelock", None),
        ("typing-extensions", None),
        ("sympy", None),
        ("mpmath", None),
        ("networkx", None),
        ("jinja2", None),
        ("MarkupSafe", None),
        ("fsspec", None),
        ("pillow", None),
        ("smplx", "0.1.28"),
        ("pyrender", None),
        ("opencv-python", None),
        ("yacs", None),
        ("scikit-image", None),
        ("scipy", None),
        ("PyWavelets", None),
        ("imageio", None),
        ("tifffile", None),
        ("lazy_loader", None),
        ("einops", None),
        ("timm", None),
        ("dill", None),
        ("pandas", None),
        ("pytz", None),
        ("python-dateutil", None),
        ("rich", None),
        ("markdown-it-py", None),
        ("mdurl", None),
        ("pygments", None),
        ("hydra-core", None),
        ("omegaconf", None),
        ("antlr4-python3-runtime", "4.9.3"),
        ("hydra-colorlog", None),
        ("colorama", None),
        ("pyrootutils", None),
        ("python-dotenv", None),
        ("webdataset", None),
        ("braceexpand", None),
        ("gdown", None),
        ("beautifulsoup4", None),
        ("soupsieve", None),
        ("requests", None),
        ("urllib3", None),
        ("certifi", None),
        ("charset-normalizer", None),
        ("idna", None),
        ("tqdm", None),
        ("pytorch-lightning", None),
        ("torchmetrics", None),
        ("lightning-utilities", None),
        ("gradio", None),
        ("pyopengl", None),
        ("pyopengl-accelerate", None),
        ("trimesh", None),
        ("fvcore", None),
        ("iopath", None),
        ("chumpy", None),
    ]
    for pkg, ver in pypi_pkgs:
        download_pypi(pkg, ver)
    
    print("ALL PACKAGES DOWNLOADED AND EXTRACTED!")
