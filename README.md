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

### ResNet20 and ResNet110 on Cifar10
This section is built on top of  Yerlan Idelbayev's wonderful repo: https://github.com/akamaster/pytorch_resnet_cifar10.
#### ResNet20 
##### Evaluate backbone:
Let's first evaluate the backbone using this cmdline
```
python trainer20.py --save_path <save_best_weight> --tb_filename <save_tensorboard_path> --pretrained_path pretrained_models/resnet20-12fca82f.th   --epochs 1 --pretrained --arch resnet20 --model resnet20 --evaluate --dataset_path <path2dataset> &  
```
*Expected Top1: 91.73*
##### Integrate block and FT:
Let's integrate the block and finetune the network
```
python trainer20.py --save_path <path> --tb_filename <tb_path> --pretrained_path pretrained_models/resnet20-12fca82f.th --epochs 200 --pretrained --num_loops 2 --arch resnet20 --model resnet20_feedback --lr 0.1 --alpha 0.0001 --original_weights pretrained_models/resnet20-12fca82f.th --dataset_path <path2dataset> 
```
##### Evaluate model
```
python trainer20.py --pretrained_path <weights2evaluate>  --epochs 200 --pretrained --num_loops 2 --arch resnet20 --model resnet20_feedback --alpha 0.0001 --original_weights pretrained_models/resnet20-12fca82f.th --dataset_path <path2dataset> --evaluate
```
A pretrained model can be evaluated:
```
python trainer20.py --pretrained_path pretrained_models/2loops_ep138_acc93.16.pth  --epochs 200 --pretrained --num_loops 2 --arch resnet20 --model resnet20_feedback --alpha 0.0001 --original_weights pretrained_models/resnet20-12fca82f.th --dataset_path <path2dataset> --evaluate
```
*Expected Top1: 93.16*

### ResNet50 on ImageNet (Coming soon)
We will build our model on top of PyTorch Resnet50.
First download the backbone weights from:
[Resnet50 Weights](https://download.pytorch.org/models/resnet50-19c8e357.pth)
#### Evaluate backbone:
Let's first evaluate PyTorch's Resnet50, expected results:76.13
```
python3 main_resnet.py --lr 0.0005 --num_loops 2  --batch-size 240 --pretrained_path resnet50-19c8e357.pth --model resnet --pretrained True --evaluate --pretrained True
```
#### Train Feedback:
```
python3 main_resnet.py --lr 0.0005 --num_loops 2  --batch-size 240 --pretrained_path res
net50-19c8e357.pth   --model resnet_feedback --pretrained True
```
#### Evaluate Feedback:
```
python3 main_resnet.py --lr 0.0005 --num_loops 2  --batch-size 240 --pretrained_path <path2weights> --model resnet_feedback --pretrained True --evaluate zx
```
