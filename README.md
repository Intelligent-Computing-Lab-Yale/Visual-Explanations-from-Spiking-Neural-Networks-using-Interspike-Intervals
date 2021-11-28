# Visual-Explanations-from-Spiking-Neural-Networks-using-Interspike-Intervals
Kim, Y. & Panda, P., Visual explanations from spiking neural networks using inter-spike intervals. Sci Rep 11, 19037 (2021). https://doi.org/10.1038/s41598-021-98448-0


## Prerequisites
* Python 3.9    
* PyTorch 1.10.0     
* NVIDIA GPU (>= 12GB)      
* CUDA 10.2 (optional)         

## Getting Started

### Conda Environment Setting
```
conda create -n VisualExp 
conda activate VisualExp
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

## Training and testing

* Arguments required for training and testing are contained in ``config.py``` 
* Here is an example of running an experiment on CIFAR100
* (if a user want to skip search process and use predefined architecgtur) A architecture can be parsed by ``--cnt_mat 0302 0030 3003 0000`` format


### Training

*  Train a model using BNTT (https://github.com/Intelligent-Computing-Lab-Yale/BNTT-Batch-Normalization-Through-Time).


### Testing (on pretrained model)

* As a first step, download pretrained parameters ([link][e]) to ```./savemodel/save_cifar100_bw.pth.tar```   

[e]: https://drive.google.com/file/d/1pnS0nFMk2KlxTFeeVT5fYMdTPh_8qn84/view?usp=sharing

* The above pretrained model is for CIFAR100 / architecture ``--cnt_mat 0302 0030 3003 0000``

*  Run the following command

```
python search_snn.py  --dataset 'cifar100' --cnt_mat 0302 0030 3003 0000 --savemodel_pth './savemodel/save_cifar100_bw.pth.tar'  --celltype 'backward'
```


 

