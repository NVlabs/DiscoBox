[![NVIDIA Source Code License](https://img.shields.io/badge/license-NSCL-blue.svg)](LICENSE)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)


**We quit maintaining this project. Please check our new work, [Mask Auto-labeler](https://github.com/NVlabs/mask-auto-labeler) for more powerful models**

## DiscoBox: Weakly Supervised Instance Segmentation and Semantic Correspondence from Box Supervision

<div style="bottom:20px; width: 00px; " align="center">
  
![output](https://user-images.githubusercontent.com/6581457/162640805-1134bd88-ce08-48da-b643-3b2fffba8611.gif)
  
</div>

### [Paper](https://arxiv.org/abs/2105.06464) | [Blog](https://developer.nvidia.com/blog/segment-objects-without-masks-and-reduce-annotation-effort-using-the-discobox-dl-framework/) | [Demo (Youtube)](https://youtu.be/fuesBLYSeu8) | [Demo (Bilibili)](https://www.bilibili.com/video/BV1dL41137oJ/)

DiscoBox: Weakly Supervised Instance Segmentation and Semantic Correspondence from Box Supervision.<br>
[Shiyi Lan](https://voidrank.github.io/), [Zhiding Yu](https://chrisding.github.io/), [Chris Choy](https://chrischoy.github.io/), [Subhashree Radhakrishnan](https://www.linkedin.com/in/subhashree-radhakrishnan-b0b0048b), [Guilin Liu](https://liuguilin1225.github.io/), [Yuke Zhu](https://www.cs.utexas.edu/~yukez/), [Larry Davis](http://users.umiacs.umd.edu/~lsd/), [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/)<br>
International Conference on Computer Vision (ICCV) 2021

This repository contains the official Pytorch implementation of training & evaluation code and pretrained models for [DiscoBox](https://arxiv.org/abs/2105.06464).
DiscoBox is a state of the art framework that can jointly predict high quality instance segmentation and semantic correspondence from box annotations.

We use [MMDetection v2.10.0](https://github.com/open-mmlab/mmdetection/tree/v2.10.0) as the codebase.

All of our models are trained and tested using [automatic mixed precision](https://pytorch.org/docs/stable/amp.html), which leverages float16 for speedup and less GPU memory consumption. 


## Installation

This implementation is based on [PyTorch==1.9.0](https://github.com/pytorch/pytorch/tree/v1.9.0), [mmcv==1.3.13](https://github.com/open-mmlab/mmcv/tree/v1.3.13), and [mmdetection==2.10.0](https://github.com/open-mmlab/mmdetection/tree/v2.10.0)


Please refer to [get_started.md](docs/get_started.md) for installation.

Or you can download the docker image from [our dockerhub repository](https://hub.docker.com/repository/docker/voidrank/discobox).


## Models

### Results on COCO val 2017

|     Backbone    | Weights |  AP  | AP@50 | AP@75 | AP@Small | AP@Medium | AP@Large |
|:---------------:|---------|:----:|:-----:|:-----:|:--------:|:---------:|:--------:|
|    ResNet-50    |     [download](https://drive.google.com/file/d/1550Osa2YpcgFFjx_y7dvjVie5jcLmsJW/view?usp=sharing)  | 30.7 |  52.6 |  30.6 |   13.3   |    34.1   |   45.6   |
|  ResNet-101-DCN |     [download](https://drive.google.com/file/d/1drOZ2fzPxgadrfuovObqS3aLzUxDQtjH/view?usp=sharing)    | 35.3 |  59.1 |  35.4 |   16.9   |    39.2   |   53.0   |
| ResNeXt-101-DCN |     [download](https://drive.google.com/file/d/1vyfdMhQkvGBHvp3LKUlUUcFcHNJU15Dv/view?usp=sharing)    | 37.3 |  60.4 |  39.1 |   17.8   |    41.1   |   55.4   |


### Results on COCO test-dev

We also evaluate the models in the section `Results on COCO val 2017` with the **same** weights on COCO test-dev.

|     Backbone    | Weights |  AP  | AP@50 | AP@75 | AP@Small | AP@Medium | AP@Large |
|:---------------:|---------|:----:|:-----:|:-----:|:--------:|:---------:|:--------:|
|    ResNet-50    |     [download](https://drive.google.com/file/d/1550Osa2YpcgFFjx_y7dvjVie5jcLmsJW/view?usp=sharing)  | 32.0 |  53.6 |  32.6 |   11.7   |    33.7   |   48.4   |
|  ResNet-101-DCN |     [download](https://drive.google.com/file/d/1drOZ2fzPxgadrfuovObqS3aLzUxDQtjH/view?usp=sharing)    | 35.8 |  59.8 |  36.4 |   16.9   |    38.7   |   52.1   |
| ResNeXt-101-DCN |     [download](https://drive.google.com/file/d/1vyfdMhQkvGBHvp3LKUlUUcFcHNJU15Dv/view?usp=sharing)    | 37.9 |  61.4 |  40.0 |   18.0   |    41.1   |   53.9   |


## Training

### COCO 

ResNet-50 (8 GPUs): 

```
bash tools/dist_train.sh \
     configs/discobox/discobox_solov2_r50_fpn_3x.py 8
```

ResNet-101-DCN (8 GPUs): 

```
bash tools/dist_train.sh \
     configs/discobox/discobox_solov2_r101_dcn_fpn_3x.py 8
```

ResNeXt-101-DCN (8 GPUs): 

```
bash tools/dist_train.sh \
     configs/discobox/discobox_solov2_x101_dcn_fpn_3x.py 8
```

### Pascal VOC 2012

ResNet-50 (4 GPUs):

```
bash tools/dist_train.sh \
     configs/discobox/discobox_solov2_voc_r50_fpn_6x.py 4
```

ResNet-101 (4 GPUs):

```
bash tools/dist_train.sh \
     configs/discobox/discobox_solov2_voc_r101_fpn_6x.py 4
```


## Testing

 
### COCO

ResNet-50 (8 GPUs): 

```
bash tools/dist_test.sh \
     configs/discobox/discobox_solov2_r50_fpn_3x.py \
     work_dirs/coco_r50_fpn_3x.pth 8 --eval segm
```

ResNet-101-DCN (8 GPUs): 

```
bash tools/dist_test.sh \
     configs/discobox/discobox_solov2_r101_dcn_fpn_3x.py \
     work_dirs/coco_r101_dcn_fpn_3x.pth 8 --eval segm
```

ResNeXt-101-DCN (GPUs): 

```
bash tools/dist_test.sh \
     configs/discobox/discobox_solov2_x101_dcn_fpn_3x_fp16.py \
     work_dirs/coco_x101_dcn_fpn_3x.pth 8 --eval segm
```


## Box-conditioned inference

You can use `DiscoBox` for autolabeling given images and tight bounding boxes. We call this box-conditioned inference. Here is an example of box-conditioned inference on COCO val2017 with `x101_dcn_fpn` arch:

```
bash tools/dist_test.sh \
     config/discobox/boxcond_discobox_solov2_x101_dcn_fpn_3x.py \
     work_dirs/x101_dcn_fpn_coco_3x.pth 8 \
     --format-only \
     --options "jsonfile_prefix=work_dirs/coco_x101_dcn_fpn_results.json"
```


### Pascal VOC 2012 (COCO API)

ResNet-50 (4 GPUs): 

```
bash tools/dist_test.sh \
     configs/discobox/discobox_solov2_voc_r50_fpn_3x_fp16.py \
     work_dirs/voc_r50_6x.pth 4 --eval segm
```

ResNet-101 (4 GPUs): 

```
bash tools/dist_test.sh \
     configs/discobox/discobox_solov2_voc_r101_fpn_3x_fp16.py \
     work_dirs/voc_r101_6x.pth 4 --eval segm
```

### Pascal VOC 2012 (Matlab)

***Step 1: generate results***

ResNet-50 (4 GPUs): 

```
bash tools/dist_test.sh \
     configs/discobox/discobox_solov2_voc_r50_fpn_3x_fp16.py \
     work_dirs/voc_r50_6x.pth 4 \
     --format-only \
     --options "jsonfile_prefix=work_dirs/voc_r50_results.json"
```

ResNet-101 (4 GPUs): 

```
bash tools/dist_test.sh \
     configs/discobox/discobox_solov2_voc_r101_fpn_3x_fp16.py \
     work_dirs/voc_r101_6x.pth 4 \
     --format-only \
     --options "jsonfile_prefix=work_dirs/voc_r101_results.json"
```

***Step 2: format conversion***

ResNet-50:

```
python tools/json2mat.py work_dirs/voc_r50_results.json work_dirs/voc_r50_results.mat
```

ResNet-101:

```
python tools/json2mat.py work_dirs/voc_r101_results.json work_dirs/voc_r101_results.mat
```

***Step 3: evaluation***

Please visit [BBTP](https://github.com/chengchunhsu/WSIS_BBTP) for the evaluation code written in Matlab.


### PF-Pascal

Please visit [this repository](https://github.com/voidrank/SCOT).

## Visualization

ResNeXt-101

```
python tools/test.py configs/discobox/discobox_solov2_x101_dcn_fpn_3x.py coco_x101_dcn_fpn_3x.pth --show --show-dir discobox_vis_x101
```

## LICENSE

Please check the LICENSE file. DiscoBox may be used non-commercially, meaning for research or 
evaluation purposes only. For business inquiries, please contact 
[researchinquiries@nvidia.com](mailto:researchinquiries@nvidia.com).

## Citation



```BibTeX
@article{lan2021discobox,
  title={DiscoBox: Weakly Supervised Instance Segmentation and Semantic Correspondence from Box Supervision},
  author={Lan, Shiyi and Yu, Zhiding and Choy, Christopher and Radhakrishnan, Subhashree and Liu, Guilin and Zhu, Yuke and Davis, Larry S and Anandkumar, Anima},
  journal={arXiv preprint arXiv:2105.06464},
  year={2021}
}
```



