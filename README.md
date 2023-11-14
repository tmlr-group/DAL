# Learning to Augment Distributions for Out-of-distribution Detection

**[Learning to Augment Distributions for Out-of-Distribution Detection](https://openreview.net/forum?id=OtU6VvXJue)**   (NeurIPS 2023)

Qizhou Wang, Zhen Fang, Yonggang Zhang, Feng Liu, Yixuan Li, and Bo Han.

**Keywords**: Out-of-distribution Detection, Reliable Machine Learning


**Abstract**: Open-world classification systems should discern out-of-distribution (OOD) data whose labels deviate from those of in-distribution (ID) cases, motivating recent studies in OOD detection. Advanced works, despite their promising progress, may still fail in the open world, owing to the lack of knowledge about unseen OOD data in advance. Although one can access auxiliary OOD data (distinct from unseen ones) for model training, it remains to analyze how such auxiliary data will work in the open world. To this end, we delve into such a problem from a learning theory perspective, finding that the distribution discrepancy between the auxiliary and the unseen real OOD data is the key to affecting the open-world detection performance. Accordingly, we propose Distributional-Augmented OOD Learning (DAL), alleviating the OOD distribution discrepancy by crafting an OOD distribution set that contains all distributions in a Wasserstein ball centered on the auxiliary OOD distribution. We justify that the predictor trained over the worst OOD data in the ball can shrink the OOD distribution discrepancy, thus improving the open-world detection performance given only the auxiliary OOD data. We conduct extensive evaluations across representative OOD detection setups, demonstrating the superiority of our DAL over its advanced counterparts. 


```
@inproceedings{
wang2023dal,
title={Learning to Augment Distributions for Out-of-distribution Detection},
author={Wang, Qizhou and Fang, Zhen and Zhang, Yonggang and Liu, Feng and Li, Yixuan and Han, Bo},
booktitle={NeurIPS},
year={2023},
url={https://openreview.net/forum?id=OtU6VvXJue}
}
```

## Get Started

### Environment
- Python (3.7.10)
- Pytorch (1.7.1)
- torchvision (0.8.2)
- CUDA
- Numpy

### Pretrained Models and Datasets

Pretrained models are provided in folder

```
./models/
```

Please download the datasets in folder

```
./data/
```

Surrogate OOD Dataset

- [80 Million Tiny Images](https://github.com/hendrycks/outlier-exposure)


Test OOD Datasets 

- [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

- [Places365](http://places2.csail.mit.edu/download.html)

- [LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)

- [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)


## Training

To train the DOE model on CIFAR benckmarks, simply run:

- CIFAR-10
```train cifar10
python main.py cifar10 --gamma=10 --beta=.01  --rho=10  --iter=10 --learning_rate=0.07 --strength=1
```


- CIFAR-100
```train cifar100
python main.py cifar100  --gamma=10 --beta=.005  --rho=10  --iter=10 --learning_rate=0.07 --strength=1
```

## Results

The key results on CIFAR benchmarks are listed in the following table. 

|     | CIFAR-10 | CIFAR-10 | CIFAR-100 | CIFAR-100 |
|:---:|:--------:|:--------:|:---------:|:---------:|
|     |   FPR95  |   AUROC  |   FPR95   |   AUROC   |
| MSP |   50.15  |   91.02  |   78.61 | 75.95   |
|  OE |   4.67  |   98.88  |   43.14 | 90.27   |
| DAL |   **2.68**   |   **99.01**  |   **29.68**   |   **93.92**   |
