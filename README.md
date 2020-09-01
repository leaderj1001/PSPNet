# PSPNet
Implementing Pyramid Scene Parsing Network (PSPNet) paper using Pytorch

  - PSPNet
  - Reference
    - [Paper](https://arxiv.org/pdf/1612.01105.pdf)
    - Author: Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia
    - Organization: The Chinese University of Hong Kong, SenseTime Group Limited
    
## Usage
  1. Download VOC2012 Dataset
  2. Data Tree
  ```
  Dataset
   ├── prepare_voc.py
   ├── VOC2012
   │   ├── Annotations
   |   ├── Images
   │   ├── ImageSets
   │   │   |   ├── Action
   │   │   |   ├── Layout
   │   │   |   ├── Main
   │   │   |   ├── Segmentation
   │   │   |   |   ├── train.txt
   │   │   |   |   ├── trainaug.txt
   │   │   |   |   ├── trainaugval.txt
   │   │   |   |   ├── trainval.txt
   │   │   |   |   ├── val.txt
   │   ├── Labels
   |   ├── SegmentationClass
   │   ├── SegmentationObject
  ```
  3. Prepare VOC2012
  ```
  python dataset/prepared_voc.py
  ```
  4. Train
  - 1 GPU
    ```
    python main.py --evaluation False
    ```
  - Multi GPUs (ex, 8 GPUs)
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=$RANDOM main.py
    ```
  5. Test
  ```
  python main.py --evaluation True
  ```

## Experiment (VOC2012, Val)
| Datasets | Resource | Model | Mean IoU | Pixel Acc
| :---: | :---: | :---: | :---: | :---: |
VOC2012 | 8 GPUs | PSPNet50 (Baseline) | 0.7802 | 0.9513 |
VOC2012 | 8 GPUs | PSPNet50 (Our) | 0.7644 | 0.9400 |
VOC2012 | 8 GPUs | PSPNet50 (Our + Skip(Out1)) | 0.77414 | 0.9431 |
VOC2012 | 8 GPUs | PSPNet50 (Our + Skip(Out1, Stem)) | 0.78141 | 0.9501 |
VOC2012 | 1 GPU | PSPNet50 (Our) | 0.7378 | 0.9322 |
VOC2012 | 8 GPUs | PSPNet101 (Baseline) | 0.7963 | 0.9550 |
VOC2012 | 8 GPUs | PSPNet101 (Our) | 0.7931 | 0.9487 |
