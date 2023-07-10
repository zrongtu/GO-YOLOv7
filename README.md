# GO-YOLOv7
The official repository for "GO-YOLOv7: A Lightweight and Fast Algorithm for Detecting Passion Fruit under Orchard Environment".

## Dataset
### 1. Introduction
Passion fruit dataset was collected from an orchard in Wutong Town, Lingui District, Guilin, Guangxi, using a Realsense D435 depth camera, and each RGB image has a resolution of 640 × 480 pixels.

LabelImg was used as the image annotation tool. When using LabelImg to mark the passion fruit, the position and classification of the passion fruit was marked.

Data augmentation on the original dataset can increase the diversity and robustness of the experimental dataset.. Considering the actual interference factors of the natural environment, several processes were used in the augmentation including rotating the original image, adjusting the image brightness, and adding noise. 

### 2. Downloads
```text
Link：https://pan.baidu.com/s/1hviW2N6dobgKTBXsfV5LpQ 
Password：4d4e 
```

## Requirements
1. Python: 3.8
2. CUDA: 11.4
3. Pytorch: 1.11.0

## Getting Started
### 1. Install
```text
git clone https://github.com/zrongtu/GO-YOLOv7.git  # clone
cd GO-YOLOv7
pip install -r requirements.txt  # install
```
### 2. Training
```text
python train.py --batch-size 128 --data data/your_dataset_name.yaml --cfg cfg/training/your_dataset_name.yaml --weights yolov7-tiny.pt --epochs 200  
```
