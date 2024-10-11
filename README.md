# ComfyUI-PMRF

### ComfyUI node for [PMRF](https://github.com/ohayonguy/PMRF)

![ComfyUI-PMRF](https://github.com/user-attachments/assets/c6692669-7335-424b-8377-b9aa85ac258c)

Install by git cloning this repo to your ComfyUI custom_nodes directory and then restarting ComfyUI, after which all the PMRF models will be downloaded and necessary packages like NATTEN will be automatically installed for you...
```
from your main ComfyUI dir:
cd custom_nodes
git clone https://github.com/2kpr/ComfyUI-PMRF
```

<br/>

PMRF uses [NATTEN](https://shi-labs.com/natten/), which requires matching its builds to your ComfyUI's venv/conda environment's CUDA and Torch versions.

This isn't a problem on linux as there are ample builds available for CUDA 11.8 to 12.4 and Torch 2.1 to 2.4, but if you are on Windows it is another story. NATTEN doesn't provide any Windows builds, just a means to [build/install NATTEN on Windows yourself](https://github.com/SHI-Labs/NATTEN/blob/main/docs/install.md#build-with-msvc) if you have MSVC and the CUDA toolkit installed, etc.

Therefore to make this ComfyUI-PMRF node a bit more accessible to Windows users I have spent the time to build 3 variants of NATTEN for Windows which are located here: [https://huggingface.co/2kpr/NATTEN-Windows](https://huggingface.co/2kpr/NATTEN-Windows). These 3 variants are for python 3.10, 3.11, and 3.12, each tied to CUDA 12.4 and Torch 2.4.

So, if you are on Windows and you meet those specs then NATTEN will be automatically installed for you from one of those 3 variant builds and there is nothing you need to do extra, but if you happen to be on Windows and have a python version below 3.10 or above 3.12 and/or you don't have both CUDA 12.4 and Torch 2.4 installed in your ComfyUI's venv/conda environment, then you will have to [build/install NATTEN on Windows yourself](https://github.com/SHI-Labs/NATTEN/blob/main/docs/install.md#build-with-msvc).

<br/>

### PMRF Links:
https://pmrf-ml.github.io/<br/>
https://github.com/ohayonguy/PMRF<br/>
https://arxiv.org/abs/2410.00418<br/>
https://huggingface.co/spaces/ohayonguy/PMRF<br/>
