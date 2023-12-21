# -*- coding: utf-8 -*-
"""
CS7GV1 Computer Vision - Create Datasets

This script splits the CelebA dataset into training, validation and test data and saves them to seperate directories.

Prerequisites: 
1. Download following files from https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg
    - Anno/identity_CelebA.txt
    - Img/img_align_celeba.zip
2. Unzip Img/img_align_celeba.zip

Example usage:
    python dataset.py --img-dir ../downloads/img_align_celeba --id-file ../downloads/identity_CelebA.txt

@author: K. Nolle
"""

import argparse
import cv2
import numpy as np
import os
import pandas as pd
import random
import shutil
from sklearn.model_selection import train_test_split

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', help='Directory with unzipped images', type=str)
    parser.add_argument('--id-file', help='File name with identities', type=str)
    parser.add_argument('--aug', help='Pass value 1 if training data should be augmented', type=int)
    
    args = parser.parse_args()
    
    if args.img_dir is None:
        print("Please specify the directory with unzipped images")
        exit(1)

    if args.id_file is None:
        print("Please specify the file name with identities")
        exit(1)

        
    train_dir = "train"
    train_dir_aug = "train_aug"
    val_dir = "validation"
    test_dir = "test"
    
    labels = 500
        
    
    # Read file with annotations
    df = pd.read_csv(args.id_file, sep=' ', names=["filename", "id"])

    # Select labels with highest number of samples
    df_agg = df.groupby("id")["id"].value_counts().sort_values(ascending=False)[:labels].reset_index()
    
    # Re-label images
    df_new = df_agg.reset_index(names="new_id")
    df = df.merge(df_new, how="inner", on="id")
    
    print("Number of labels:", len(pd.unique(df["new_id"])))
    print("Number of total samples:", len(df))
    

    # Split dataset into 20% test and 80% training data
    filenames_train, filenames_test, id_train, id_test = train_test_split(np.array(df["filename"]), np.array(df["new_id"]), 
                                                                          stratify=np.array(df["new_id"]), test_size=0.2, random_state=42)
    
    # Split training data again into 20% validation and 80% training data
    filenames_train, filenames_val, id_train, id_val = train_test_split(filenames_train, id_train, stratify=id_train, test_size=0.2, random_state=123)

    if args.aug is not None and args.aug == 1:    
        # Create missing directory
        if not os.path.exists(train_dir_aug):
            print("Creating output directory " + train_dir_aug)
            os.makedirs(train_dir_aug)
        
        # Copy images in training set to directory and encode id in file name
        # Also augment images randomly and save to directiry
        print(f"Copying and augmenting training data ({len(filenames_train)} files)...")
        for i in range(len(filenames_train)):
            filename = filenames_train[i]
            label = id_train[i]
                
            # Read image
            raw_data = cv2.imread(os.path.join(args.img_dir, filename))
            
            # Save original image
            cv2.imwrite(os.path.join(train_dir_aug, filename.split('.')[0]+f"_0_id-{label}.jpg"), raw_data)
            
            for i in range(1, 5):
                aug_img = raw_data
                
                # Randomly flip along y-axis
                random.seed(int(label)+i)
                p = random.random()
                if p < 0.5:
                    aug_img = cv2.flip(aug_img, 1)
                
                # Randomly gamma correct
                invGamma = 1.0 / random.uniform(0.25, 2)
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                aug_img = cv2.LUT(aug_img, table)
                
                # Randomly rotate image
                x = random.randint((aug_img.shape[1]/2)-25, (aug_img.shape[1]/2)+25)  # Random rotation center
                y = random.randint((aug_img.shape[0]/2)-25, (aug_img.shape[0]/2)+25)
                angle = random.uniform(-30, 30)  # Random angle
                M = cv2.getRotationMatrix2D(center=(x, y), angle=angle, scale=1) # Rotation matrix
                aug_img = cv2.warpAffine(aug_img, M=M, dsize=(aug_img.shape[1], aug_img.shape[0]))
                
                cv2.imwrite(os.path.join(train_dir_aug, filename.split('.')[0]+f"_{i}_id-{label}.jpg"), aug_img)
    else:
        # Create missing directory
        if not os.path.exists(train_dir):
            print("Creating output directory " + train_dir)
            os.makedirs(train_dir)
            
        # Copy images in training set to directory and encode id in file name
        print(f"Copying training data ({len(filenames_train)} files)...")
        for i in range(len(filenames_train)):
            shutil.copyfile(os.path.join(args.img_dir, filenames_train[i]), 
                            os.path.join(train_dir, filenames_train[i].split('.')[0]+"_id-"+str(id_train[i])+".jpg"))
         
            
    # Create missing directory  
    if not os.path.exists(val_dir):
        print("Creating output directory " + val_dir)
        os.makedirs(val_dir)
        
    # Copy images in training set to directory and encode id in file name
    print(f"Copying validation data ({len(filenames_val)} files)...")
    for i in range(len(filenames_val)):
        shutil.copyfile(os.path.join(args.img_dir, filenames_val[i]), 
                        os.path.join(val_dir, filenames_val[i].split('.')[0]+"_id-"+str(id_val[i])+".jpg"))
    
    
    # Create missing directory
    if not os.path.exists(test_dir):
        print("Creating output directory " + test_dir)
        os.makedirs(test_dir)
        
    # Copy images in test set to directory and encode id in file name
    print(f"Copying test data ({len(filenames_test)} files)...")
    for i in range(len(filenames_test)):
        shutil.copyfile(os.path.join(args.img_dir, filenames_test[i]), 
                        os.path.join(test_dir, filenames_test[i].split('.')[0]+"_id-"+str(id_test[i])+".jpg"))


 
if __name__ == '__main__':
    main()