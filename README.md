![图片](https://github.com/user-attachments/assets/fdb36dd9-41ef-49a4-94ac-0459e69bf1b5)



This paper presents an enhanced model, called CCCNet, which incorporates an efficient global attention module, known as the Criss-Cross module, into the Feature Pyramid Network structure of the CLRNet.

### Prerequisites

The training and testing environment consists of the following specifications: 

Ubuntu version 18.04.6 LTS, 

CUDA version 11.3, 

cuDNN version 8.2.0.5, 

Python version 3.8.18, 

PyTorch version 1.8.0.

Other dependencies described in requirements.txt


### Clone this repository

Clone this code to your workspace. We call this directory as $CLRNET_ROOT

git clone https://github.com/grass2440/CCCNet.git



### Create a conda virtual environment and activate it (conda is optional)

conda create -n clrnet python=3.8 -y

conda activate clrnet



### Data preparation
#### CULane

Download CULane. Then extract them to $CULANEROOT. Create link to data directory.
```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```
For CULane, you should have structure like this:
```Shell
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```
#### Tusimple

Download Tusimple. Then extract them to $TUSIMPLEROOT. Create link to data directory.
```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $TUSIMPLEROOT data/tusimple
```
For Tusimple, you should have structure like this:
```Shell
$TUSIMPLEROOT/clips # data folders
$TUSIMPLEROOT/lable_data_xxxx.json # label json file x3
$TUSIMPLEROOT/test_tasks_0627.json # test tasks json file
```
For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation.
```Shell
python tools/generate_seg_tusimple.py --root $TUSIMPLEROOT # this will generate seg_label directory
```


### Training

For training, run
```Shell
python main.py [configs/path_to_your_config] --gpus [gpu_num]
```
For example, run
```Shell
python main.py configs/clrnet/clr_resnet18_culane.py --gpus 0
```


### Validation

For testing, run
```Shell
python main.py [configs/path_to_your_config] --[test|validate] --load_from [path_to_your_model] --gpus [gpu_num]
```
For example, run
```Shell
python main.py configs/clrnet/clr_dla34_culane.py --validate --load_from culane_dla34.pth --gpus 0
```
Currently, this code can output the visualization result when testing, just add `--view`. We will get the visualization result in `work_dirs/xxx/xxx/visualization`.



### Citation
```
@InProceedings{Zheng_2022_CVPR,

    author    = {Zheng, Tu and Huang, Yifei and Liu, Yang and Tang, Wenjian and Yang, Zheng and Cai, Deng and He, Xiaofei},
    title     = {CLRNet: Cross Layer Refinement Network for Lane Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {898-907}
}
```
```
@article{huang2020ccnet,
  author={Huang, Zilong and Wang, Xinggang and Wei, Yunchao and Huang, Lichao and Shi, Humphrey and Liu, Wenyu and Huang, Thomas S.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={CCNet: Criss-Cross Attention for Semantic Segmentation}, 
  year={2020},
  month={},
  volume={},
  number={},
  pages={1-1},
  keywords={Semantic Segmentation;Graph Attention;Criss-Cross Network;Context Modeling},
  doi={10.1109/TPAMI.2020.3007032},
  ISSN={1939-3539}}
```
```
@article{huang2018ccnet,
    title={CCNet: Criss-Cross Attention for Semantic Segmentation},
    author={Huang, Zilong and Wang, Xinggang and Huang, Lichao and Huang, Chang and Wei, Yunchao and Liu, Wenyu},
    booktitle={ICCV},
    year={2019}}
```





















