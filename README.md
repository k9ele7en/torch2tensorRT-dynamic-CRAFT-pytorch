# Convert Pytorch pretrain -> TensoRT engine directly for CRAFT (Character-Region Awareness For Text detection)
- Convert CRAFT Text detection pretrain Pytorch model into TensorRT engine directly, without ONNX step between<br>
- CRAFT: (forked from https://github.com/clovaai/CRAFT-pytorch)
Official Pytorch implementation of CRAFT text detector | [Paper](https://arxiv.org/abs/1904.01941) | [Pretrained Model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) | [Supplementary](https://youtu.be/HI8MzpY8KMI)
- Using torch2trt_dynamic from https://github.com/grimoire/torch2trt_dynamic (branch of https://github.com/NVIDIA-AI-IOT/torch2trt with dynamic shapes support)

### Overview
Implementation of convenient converter from Pytorch to Tensor RT directly for CRAFT text detector.
This repo is only about converting Pytorch model into Tensor RT. For Tensor RT inference, check out:
- Inference using Tensor RT standalone (https://github.com/k9ele7en/ONNX-TensorRT-Inference-CRAFT-pytorch)
- Advance inference pipeline using NVIDIA Triton Server (https://github.com/k9ele7en/triton-tensorrt-CRAFT-pytorch)

### Author
k9ele7en. Give 1 star if you find some value in this repo. <br>
Thank you.
### License
[MIT License] A short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source.

## Updates
**7 Aug, 2021**: Initial repo, converter run success.


## Getting started
### 1. Install dependencies
#### Requirements
```
$ pip install -r requirements.txt
```
#### Install ONNX, TensorRT
Check details at [./README_Env.md](./README_Env.md)

### 2. Download the trained models
 
 *Model name* | *Used datasets* | *Languages* | *Purpose* | *Model Link* |
 | :--- | :--- | :--- | :--- | :--- |
General | SynthText, IC13, IC17 | Eng + MLT | For general purpose | [Click](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
IC15 | SynthText, IC15 | Eng | For IC15 only | [Click](https://drive.google.com/open?id=1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf)
LinkRefiner | CTW1500 | - | Used with the General Model | [Click](https://drive.google.com/open?id=1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO)

### 3. Start converting Pytorch->TensorRT
```
$ cd converters
$ python torch2trt.py
INFO - Converting CRAFT Pytorch pth to TensorRT engine... (torch2trt.py:31)
...
INFO - Compare Pytorch output vs TensorRT engine output...
...
INFO - Convert CRAFT Pytorch pth to tensorRT engine sucess (torch2trt.py:83)
```
