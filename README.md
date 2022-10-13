# PFLD_GhostOne: PFLD+GhostNet+MobileOne

## An ultralight face landmark detector based on PFLD, GhostNet and MobileOne
This project supplies a better face landmark detector that is suitable for embedding devices: ***PFLD_GhostOne*** which is more suitable for edge computing. In most cases, its accuracy is better than the original PFLD model, and its speed is around 55% faster than the original PFLD model.

The proposed ***GhostOne*** module is show below:
![image](https://github.com/AnthonyF333/PFLD_GhostOne/blob/main/img/6.png)
More details about GhostOne can be found on this [blog](https://blog.csdn.net/u010892804/article/details/127264664?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22127264664%22%2C%22source%22%3A%22u010892804%22%7D)

&nbsp;

## Implementation
* Support training, testing and ONNX inference of PFLD_GhostOne model
* Support detecting 98 facial landmarks
* Support exporting to ONNX format

![image](https://github.com/AnthonyF333/PFLD_GhostOne/blob/main/img/outpy.gif)
![image](https://github.com/AnthonyF333/PFLD_GhostOne/blob/main/img/nice.gif)

&nbsp;

## Dependencies
* Ubuntu 18.04
* Python 3.9
* Pytorch 1.11.0
* CUDA 11.3

&nbsp;

## WFLW Test Result

Model input size is 112x112

Model|NME|OpenVino Latency(ms)|NCNN Latency(ms)|ONNX Model Size(MB)
:--:|:--:|:--:|:--:|:--:
PFLD|0.05438|1.65(CPU)&emsp;2.78(GPU)|5.4(CPU)&emsp;5.1(GPU)|4.66
PFLD-GhostNet|0.05347|1.79(CPU)&emsp;2.55(GPU)|2.9(CPU)&emsp;5.3(GPU)|3.09
PFLD-GhostNet-Slim|0.05410|2.11(CPU)&emsp;2.54(GPU)|2.7(CPU)&emsp;5.2(GPU)|2.83
PFLD-GhostOne|0.05207|1.79(CPU)&emsp;2.18(GPU)|2.4(CPU)&emsp;5.0(GPU)|2.71

The latency is the average time of running 1000 times on 11th Gen Intel(R) Core(TM) i5-11500

### Model Zoo
Model|Pretrained Model
:--:|:--:
PFLD|[pfld_best.pth](https://drive.google.com/file/d/1tZ1RMe8a4lJ2LcEMZ8iiy4e0zTIiyJgW/view?usp=sharing)
PFLD-GhostNet|[pfld_ghostnet_best.pth](https://drive.google.com/file/d/14iTrzm2OjKVm0Ztl072gr-Ak_MjyQZm9/view?usp=sharing)
PFLD-GhostNet-Slim|[pfld_ghostnet_slim_best.pth](https://drive.google.com/file/d/1y3JX2uMVEz9MkOoGc_-s5LVQmiFQxdBr/view?usp=sharing)
PFLD-GhostOne|[pfld_ghostone_best.pth](https://drive.google.com/file/d/1kCohoj4v9KNDBSs7FqOx5lO09fuoMupI/view?usp=sharing)

&nbsp;

## Installation
**Clone and install:**
* git clone https://github.com/AnthonyF333/PFLD_GhostOne.git
* cd ./PFLD_GhostOne
* pip install -r requirement.txt
* Pytorch version 1.11.0 are needed.
* Codes are based on Python 3.9

**Data:**
* Download WFLW dataset: 
  * [WFLW.zip](https://drive.google.com/file/d/1XOcAi1bfYl2LUym0txl_A4oIRXA_2Pf1/view?usp=sharing)
* Move the WFLW.zip to ./data/ directory and unzip the WFLW.zip
* Run SetPreparation.py to generate the training and test data.
　　
* By default, it repeats 80 times for every image for augmentation, and save images in ./data/test_data_repeat80/ and ./data/train_data_repeat80/ directory.

&nbsp;

## Training
Before training, check or modify network configuration (e.g. batch_size, epoch and steps etc..) in config.py.
  * MODEL_TYPE: you can choose PFLD, PFLD_GhostNet or PFLD_GhostNet_Slim or PFLD_GhostOne for different network.
  * TRAIN_DATA_PATH: the path of training data, by default it is ./data/train_data_repeat80/list.txt which is generate by SetPreparation.py.
  * VAL_DATA_PATH: the path of validation data, by default it is ./data/test_data_repeat80/list.txt which is generate by SetPreparation.py.

After modify the configuration, run train.py to start training.

&nbsp;

## Export to ONNX
Modify the model_type and model_path in pytorch2onnx.py, and then run pytorch2onnx.py to generate the xxx.onnx file. It will optimize and simplify the ONNX file.

&nbsp;

## ONNX inference
Before test the model, modify the configuration in onnx_inference.py, include the pfld_onnx_model, test_folder, save_result_folder etc.
Then run onnx_inference.py to detect and align the images in the *test_folder*, and save results in *save_result_folder* directory.

&nbsp;&nbsp;

## References
* [PFLD](https://github.com/polarisZhao/PFLD-pytorch)
* [GhostNet](https://github.com/huawei-noah/ghostnet)
* [MobileOne](https://github.com/apple/ml-mobileone)
* [PFLD_GhostNet](https://github.com/AnthonyF333/FaceLandmark_PFLD_UltraLight)
