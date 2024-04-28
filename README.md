
# Image Super Resolution Using SR-GANs and SR-ResNet

## Problem Statement
The problem of image super-resolution (SR) involves the process of transforming a low-resolution image into a high-resolution counterpart. This task is inherently ill-posed, as the same low-resolution image can correspond to multiple high-resolution images. The challenge is to generate a high-resolution image that is as close as possible to the original, high-resolution image, based on the limited information present in the low-resolution input.

## Methodology
We have used two different approaches to tackle this problem. First of them being SR-ResNet (Super Resolution ResNet) and second being SR-GANs (Super Resoltion GANs). Since it is a task of generating the unknown pixels in the lower resolution image to generate a higher resolution image therefore we have used GANs since we know it's ability to give excellent results in generative tasks.


## Contents
The repository Consists of two different directories and a jupyter notebook namely:-
- SR-GAN Method
- SR-ResNet Method
- cv-project_Super-Resolution.ipynb

These directories contain all the files related to the specific methods respectively. Furthermore, jupyter notebook consists of results of the both the methods.


## Training
In order to train models of any of the methods you need to first update the path to the dataset in their respective DataLoaders.py file.

```
"C:/Users/Harsh Soni/Downloads/CV Project/dataset/train" --> 'Path to dataset'
"C:/Users/Harsh Soni/Downloads/CV Project/dataset/val" --> 'Path to dataset'
```

After you have updated the path to the dataset, you need to run the below mentioned command to train the respective model.

```
python Train.py
```
This code will start the training for the model for that respective method and will store the weights of the models inside a folder in the respective method's directory with *.pt as the extension for the weights.

e.g.,
```
'checkpoints_gan/checkpoint_gan.pt' --> For SR-GAN
'checkpoints_resnet/checkpoint_SR-ResNet.pt' --> For SR-ResNet
```


## Inference
In-order to infer the results of the methods, you need to replace the below mentioned path of the weights to the absolute path of the weights in your system in the file Inference.py for the respective methods.

```
checkpoint_path = 'path_to_trained_weights_respective_method'
```

To run the infernce you'll need to use the following command,
```
python Inference.py
```
This will generate an image containing the visual results on you screen.

### Results of SR-ResNets
<img src="SR-ResNet Method/Results.png" >

### Results of SR-GANs
<img src="SR-GAN Method/Results.png" >

### References
- [Research Paper](https://arxiv.org/pdf/1609.04802)

## Authors
- [Harsh Soni](https://www.linkedin.com/in/harsh-soni-1a3682228/)
- [Tanishq Kankerwal](https://www.linkedin.com/in/tanishq-kankerwal-765078230/)
- [Prince Verma](https://www.linkedin.com/in/prince-verma-43498b228/)
