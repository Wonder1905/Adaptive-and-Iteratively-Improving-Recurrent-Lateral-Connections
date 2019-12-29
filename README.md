# Adaptive-and-Iteratively-Improving-Recurrent-Lateral-Connections
An official Pytorch implementation of "Adaptive and Iteratively Improving Recurrent Lateral Connections" https://arxiv.org/abs/1910.11105 <p align="center">
<img src="BasicFeedback.png" alt="smiley" height="350px" width="600px"/>
</p>  

## Prerequisites
- ubuntu18.04
- python 3.6
- torch==1.2
- torchvision==0.4
- numpy==1.17.4
- cv2
- tensorboard 
  
## Results  
We expiremented our method on two tasks, five different datasets and six models, in this repo we will show how to reproduce our on Imagenet using Resnet50,Resnet110,Resnet20 (which will appear in the updated version of the paper) and MultiFiberNet2D.

### ResNet50 on ImageNet
We will build our model on top of PyTorch Resnet50.
First download the backbone weights from:
[Resnet50 Weights](https://download.pytorch.org/models/resnet50-19c8e357.pth)
#### Evaluate backbone:
Let's first evaluate PyTorch's Resnet50, expected results:76.13
```
python3 main_resnet.py --lr 0.0005 --num_loops 2  --batch-size 240 --pretrained_path resnet50-19c8e357.pth --model resnet --pretrained True --evaluate --pretrained True
```
####Train Feedback:
```
python3 main_resnet.py --lr 0.0005 --num_loops 2  --batch-size 240 --pretrained_path res
net50-19c8e357.pth   --model resnet_feedback --pretrained True
```
####Evaluate Feedback:
```
python3 main_resnet.py --lr 0.0005 --num_loops 2  --batch-size 240 --pretrained_path <path2weights> --model resnet_feedback --pretrained True --evaluate zx
```
