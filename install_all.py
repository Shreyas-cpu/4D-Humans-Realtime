import os
import sys
import subprocess
import glob
import zipfile

os.makedirs("wheels", exist_ok=True)
os.makedirs("pkgs", exist_ok=True)

print("Starting wheel downloads...")

# 1. Download PyTorch + CUDA wheels
torch_packages = [
    "torch==2.5.1+cu121",
    "torchvision==0.20.1+cu121",
    "numpy<2.0.0",
]
cmd = [
    sys.executable, "-m", "pip", "download",
    "--dest", "./wheels",
    "--index-url", "https://download.pytorch.org/whl/cu121",
    "--extra-index-url", "https://pypi.org/simple",
    "--python-version", "3.10",
    "--platform", "manylinux2014_x86_64",
    "--platform", "linux_x86_64",
    "--platform", "any",
    "--only-binary=:all:",
] + torch_packages

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

# 2. Download remaining pip dependencies
pip_packages = [
    "smplx==0.1.28",
    "pyrender",
    "opencv-python",
    "yacs",
    "scikit-image",
    "einops",
    "timm",
    "dill",
    "pandas",
    "rich",
    "hydra-core",
    "hydra-submitit-launcher",
    "hydra-colorlog",
    "pyrootutils",
    "webdataset",
    "gdown",
    "pytorch-lightning",
    "gradio",
    "pyopengl",
    "pyopengl-accelerate",
    "trimesh",
    "fvcore",
    "iopath",
    "pytube",
    "chumpy",
]

cmd2 = [
    sys.executable, "-m", "pip", "download",
    "--dest", "./wheels",
    "--python-version", "3.10",
    "--platform", "manylinux2014_x86_64",
    "--platform", "linux_x86_64",
    "--platform", "any",
    "--only-binary=:all:",
] + pip_packages

print("Running:", " ".join(cmd2))
subprocess.run(cmd2, check=True)

print("Extracting all wheels into pkgs/...")
wheels = glob.glob("wheels/*.whl")
print(f"Found {len(wheels)} wheels.")
for w in wheels:
    try:
        with zipfile.ZipFile(w, 'r') as zip_ref:
            zip_ref.extractall("pkgs")
        print(f"Extracted: {os.path.basename(w)}")
    except Exception as e:
        print(f"Error extracting {w}: {e}")

print("All dependencies successfully extracted into pkgs/!")
