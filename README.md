# HeyGen Video Interpolation Test Task

## Overview

This repository contains a solution for creating an output video by combining original footage and interpolated segments. The key steps for constructing the output video are:

1. **First 5 seconds**: The 5 seconds of video immediately **preceding** Frame A in the original video.  
2. **Intermediate 5 seconds**:  
   - A clip (of length `time_indent` seconds) right after Frame A,  
   - An interpolated segment of duration `time_interp`,  
   - A clip (of length `5 - time_indent - time_interp` seconds) **preceding** Frame B in the original video.  
3. **Last 5 seconds**: The 5 seconds of video **following** Frame B in the original video.

### Important Parameters

- **`time_indent`**: This parameter is selected to ensure the transition between Frame A-adjacent footage and Frame B-adjacent footage is smooth, minimizing the difficulty of interpolation. The optimal value found is time_indent = 0. More details about this selection process are available in find_indent.ipynb.
- **`time_interp`**: Controls the length of the interpolated segment. Through experimentation, using 3 frames (which is 3/25 of a second) produces a smooth interpolation.

## Downloading the Input Video and Checkpoints

```bash
wget https://resource2.heygen.ai/video/5de50af7182f462b98178064f808d7d8/1280x720.mp4 -O input.mp4

mkdir -p ./ckpt
wget https://huggingface.co/MCG-NJU/VFIMamba_ckpts/resolve/main/ckpt/VFIMamba.pkl -O ./ckpt/VFIMamba.pkl
```

## Installation

**Requirements:**

- NVIDIA driver 550.120 (or newer)
- CUDA 12.4
- 8 GB GPU memory

**Set up a virtual environment and install dependencies:**

```bash
conda create -n heygen_test python=3.10
conda activate myeheygen_testnv
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
pip install -r requirements.txt
```

## Usage

After setting up the environment, run the following command to produce the final interpolated video:

```bash
python3 interpolate_video.py --input ./original.mp4 --output ./output.mp4
```

## Validation

To verify that the final video meets the requirements (i.e., Frames A and B appear at the desired times), use the provided Jupyter notebook:

```
demo.ipynb
```

## Credits

The video frame interpolation logic is based on the VFIMamba model: https://github.com/MCG-NJU/VFIMamba
