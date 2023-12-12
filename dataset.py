# -*- coding: utf-8 -*-
"""
CS7GV1 Computer Vision - Create Datasets

This script the CelebA dataset into training, validation and test data and 
saves them to seperate directories.

Prerequisites: 
1. Download following files from https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg
    - Anno/identity_CelebA.txt
    - Img/img_align_celeba.zip
2. Unzip Img/img_align_celeba.zip

Example usage:
    python dataset.py --train-dir train --val-dir validation --test-dir test --img-dir ../downloads/img_align_celeba --id-file ../downloads/identity_CelebA.txt

@author: K. Nolle
"""

import argparse
import numpy as np
import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', help='Directory with unzipped images', type=str)
    parser.add_argument('--id-file', help='File name with identities', type=str)
    parser.add_argument('--train-dir', help='Directory to store training data', type=str)
    parser.add_argument('--val-dir', help='Directory to store validation data', type=str)
    parser.add_argument('--test-dir', help='Directory to store test data', type=str)
    args = parser.parse_args()
    
    if args.img_dir is None:
        print("Please specify the directory with unzipped images")
        exit(1)

    if args.id_file is None:
        print("Please specify the file name with identities")
        exit(1)

    if args.train_dir is None:
        print("Please specify the directory to store training data")
        exit(1)

    if args.val_dir is None:
        print("Please specify the directory to store validation data")
        exit(1)

    if args.test_dir is None:
        print("Please specify the directory to store test data")
        exit(1)
        
        
    # Create missing directories
    if not os.path.exists(args.train_dir):
        print("Creating output directory " + args.train_dir)
        os.makedirs(args.train_dir)
        
    if not os.path.exists(args.val_dir):
        print("Creating output directory " + args.val_dir)
        os.makedirs(args.val_dir)
        
    if not os.path.exists(args.test_dir):
        print("Creating output directory " + args.test_dir)
        os.makedirs(args.test_dir)
        
    # Read file with annotations
    df = pd.read_csv(args.id_file, sep=' ', names=["filename", "id"])

    # Remove labels with less than 10 samples 
    df["count"] = df.groupby("id")["id"].transform("count")
    df = df[df["count"] >= 10]
    
    print("Number of labels:", len(pd.unique(df["id"])))
    
    # Split dataset into 20% test and 80% training data
    filenames_train, filenames_test, id_train, id_test = train_test_split(np.array(df["filename"]), np.array(df["id"]), 
                                                                          stratify=np.array(df["id"]), test_size=0.2, random_state=42)
    
    # Split training data again into 20% validation and 80% training data
    filenames_train, filenames_val, id_train, id_val = train_test_split(filenames_train, id_train, stratify=id_train, test_size=0.2, random_state=123)

    # Copy images in training set to directory and encode id in file name
    print(f"Copying training data ({len(filenames_train)} files)...")
    for i in range(len(filenames_train)):
        shutil.copyfile(os.path.join(args.img_dir, filenames_train[i]), 
                        os.path.join(args.train_dir, filenames_train[i].split('.')[0]+"_id-"+str(id_train[i])+".jpg"))
        
    # Copy images in training set to directory and encode id in file name
    print(f"Copying validation data ({len(filenames_val)} files)...")
    for i in range(len(filenames_val)):
        shutil.copyfile(os.path.join(args.img_dir, filenames_val[i]), 
                        os.path.join(args.val_dir, filenames_val[i].split('.')[0]+"_id-"+str(id_val[i])+".jpg"))
    
    # Copy images in test set to directory and encode id in file name
    print(f"Copying test data ({len(filenames_test)} files)...")
    for i in range(len(filenames_test)):
        shutil.copyfile(os.path.join(args.img_dir, filenames_test[i]), 
                        os.path.join(args.test_dir, filenames_test[i].split('.')[0]+"_id-"+str(id_test[i])+".jpg"))
    
 
if __name__ == '__main__':
    main()