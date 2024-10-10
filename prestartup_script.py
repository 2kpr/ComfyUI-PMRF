import pkg_resources
import subprocess
import sys
import huggingface_hub
import importlib.util
import importlib.metadata
import folder_paths
import os
import pathlib
from packaging.version import Version
import time

pmrf_path = os.path.join(folder_paths.models_dir, "pmrf")
pmrf_model_path = os.path.join(pmrf_path, "model.safetensors")
pmrf_model_json_path = os.path.join(pmrf_path, "config.json")
if not (os.path.exists(pmrf_model_path) and os.path.exists(pmrf_model_json_path)):
    print("Downloading PMRF model from ohayonguy/PMRF_blind_face_image_restoration...")
    if not os.path.exists(pmrf_path):
        os.makedirs(pmrf_path)
    huggingface_hub.snapshot_download(
        repo_id="ohayonguy/PMRF_blind_face_image_restoration",
        local_dir=pmrf_path,
    )
upscale_models_path = os.path.join(folder_paths.models_dir, "upscale_models")
models = ["RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth"]
for model in models:
    realesrgan_path = os.path.join(upscale_models_path, model)
    if not os.path.exists(realesrgan_path):
        print(f"Downloading {model} model from 2kpr/Real-ESRGAN...")
        huggingface_hub.snapshot_download(
            repo_id="2kpr/Real-ESRGAN",
            allow_patterns=model,
            local_dir=upscale_models_path,
        )

packages = [
    {"name": "realesrgan", "version": "0.2.5"},
    {"name": "torchvision", "version": "0.19.0"},
    {"name": "torch_fidelity", "version": "0.3.0"},
    {"name": "torch_ema", "version": "0.3"},
    {"name": "pytorch_lightning", "version": "2.4.0"},
    {"name": "timm", "version": "1.0.7"},    
]

for package in packages:
    if importlib.util.find_spec(package["name"]):
        #print(f'Found package {package["name"]}')
        #print(f'Version: {package["version"]}')
        #print(f'Version: {importlib.metadata.version(package["name"])}')
        if Version(package["version"]) > Version(importlib.metadata.version(package["name"])):
            print(f'Updating {package["name"]} for PMRF...')
            subprocess.check_call([sys.executable, "-m", "pip", "install", f'{package["name"]}>={package["version"]}', "--upgrade"])
    else:
        print(f'Installing {package["name"]} for PMRF...')
        subprocess.check_call([sys.executable, "-m", "pip", "install", f'{package["name"]}>={package["version"]}', "--upgrade"])

if importlib.util.find_spec("basicsr"):
    path = pathlib.Path(importlib.util.find_spec("basicsr").origin).parent.joinpath("data/degradations.py")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if "from torchvision.transforms.functional_tensor import rgb_to_grayscale" in content:
            print(f"Patching basicsr with fix from https://github.com/XPixelGroup/BasicSR/pull/650 for PMRF...")
            content = content.replace(
                "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
                "from torchvision.transforms.functional import rgb_to_grayscale",
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

if not importlib.util.find_spec("natten"):
    print(f'Installing natten for PMRF...')
    cuda_version = ""
    torch_version = ""
    print("Searching for CUDA and Torch versions for installing atten needed by PMRF...")
    for p in pkg_resources.working_set:
        if p.project_name.startswith("nvidia-cuda-runtime"):
            if p.version.startswith("12.4"):
                cuda_version = "cu124"
                print("- Found CUDA 12.4")
            elif p.version.startswith("12.1"):
                cuda_version = "cu121"
                print("- Found CUDA 12.1")
            elif p.version.startswith("11.8"):
                cuda_version = "cu118"
                print("- Found CUDA 11.8")
        elif p.project_name == "torch":
            if p.version.startswith("2.4"):
                torch_version = "torch240"
                print("- Found Torch 2.4")
            elif p.version.startswith("2.3"):
                torch_version = "torch230"
                print("- Found Torch 2.3")
            elif p.version.startswith("2.2"):
                torch_version = "torch220"
                print("- Found Torch 2.2")
            elif p.version.startswith("2.1"):
                torch_version = "torch210"
                print("- Found Torch 2.1")
    if cuda_version == "":
        py_path = os.path.join(folder_paths.temp_directory, "torchcudaversion.py")
        if not os.path.exists(py_path):
            if not os.path.exists(folder_paths.temp_directory):
                os.makedirs(folder_paths.temp_directory)
            with open(py_path, "w", encoding="utf-8") as f:
                f.write("import torch\nprint(torch.version.cuda)")
            cuda_version = subprocess.check_output([sys.executable, f"{py_path}"]).decode().strip()
        if cuda_version == "12.4":
            cuda_version = "cu124"
            print("- Found CUDA 12.4")
        elif cuda_version == "12.1":
            cuda_version = "cu121"
            print("- Found CUDA 12.1")
        elif cuda_version == "11.8":
            cuda_version = "cu118"
            print("- Found CUDA 11.8")
    if cuda_version == "":
        print("************************************")
        print("Error: Can't find CUDA runtime version, can't install natten")
        print("       PMRF will not work until natten is installed, see https://github.com/SHI-Labs/NATTEN for help in installing natten.")
        print("************************************")
        time.sleep(4)
    elif torch_version == "":
        print("************************************")
        print("Error: Can't find torch version, can't install natten")
        print("       PMRF will not work until natten is installed, see https://github.com/SHI-Labs/NATTEN for help in installing natten.")
        print("************************************")
        time.sleep(4)
    elif cuda_version == "cu124" and torch_version != "torch240":
        print("************************************")
        print("Error: Can't install natten, which is needed by PMRF since CUDA runtime version is 12.4 but torch is not version 2.4")
        print("       PMRF will not work until natten is installed, see https://github.com/SHI-Labs/NATTEN for help in installing natten.")
        print("************************************")
        time.sleep(4)
    elif os.name == "nt" and cuda_version != "cu124":
        print("************************************")
        print("Error: Can't install natten on windows if CUDA runtime version is not 12.4 unless you build natten yourself, see https://github.com/SHI-Labs/NATTEN/blob/main/docs/install.md#build-with-msvc")
        print("       PMRF will not work until natten is installed, see https://github.com/SHI-Labs/NATTEN for help in installing natten.")
        print("************************************")
        time.sleep(4)
    elif os.name == "nt" and torch_version != "torch240":
        print("************************************")
        print("Error: Can't install natten on windows if torch version is not 2.4 unless you build natten yourself, see https://github.com/SHI-Labs/NATTEN/blob/main/docs/install.md#build-with-msvc")
        print("       PMRF will not work until natten is installed, see https://github.com/SHI-Labs/NATTEN for help in installing natten.")
        print("************************************")
        time.sleep(4)
    elif os.name == "nt" and (sys.version_info[1] < 10 or sys.version_info[1] > 12):
        print("************************************")
        print("Error: Can't install natten on windows if python version isn't 3.10, 3.11, or 3.12, unless you build natten yourself, see https://github.com/SHI-Labs/NATTEN/blob/main/docs/install.md#build-with-msvc")
        print("       PMRF will not work until natten is installed, see https://github.com/SHI-Labs/NATTEN for help in installing natten.")
        print("************************************")
        time.sleep(4)
    elif os.name == "nt":
        if sys.version_info[1] == 10:
            whl = "natten-0.17.2.dev0-py310-none-win_amd64.whl"
        elif sys.version_info[1] == 11:
            whl = "natten-0.17.2.dev0-py311-none-win_amd64.whl"
        elif sys.version_info[1] == 12:
            whl = "natten-0.17.2.dev0-py312-none-win_amd64.whl"
        whl_path = os.path.join(folder_paths.temp_directory, whl)
        if not os.path.exists(whl_path):
            if not os.path.exists(folder_paths.temp_directory):
                os.makedirs(folder_paths.temp_directory)
            print(f"Downloading {whl} from 2kpr/NATTEN-Windows...")
            huggingface_hub.snapshot_download(
                repo_id="2kpr/NATTEN-Windows",
                allow_patterns=whl,
                local_dir=folder_paths.temp_directory,
            )
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{whl_path}"])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"natten==0.17.1+{torch_version}{cuda_version}", "-f", "https://shi-labs.com/natten/wheels/"])