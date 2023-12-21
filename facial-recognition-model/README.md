# Facial Recognition Model

This folder contains the code used to create the facial recognition models for evaluating the image masks.

## Pre-Requisites: 
1. Download following files from https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg
    - Anno/identity_CelebA.txt
    - Img/img_align_celeba.zip
2. Unzip Img/img_align_celeba.zip

## Usage
- Run dataset.py to split training/validation/test data
- Run train.py to train the facial recognition model
- Run predict.py or predict_compare.py to predict labels and output accuracy of the model
