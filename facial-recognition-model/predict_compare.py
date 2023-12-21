# -*- coding: utf-8 -*-
"""
CS7GV1 Computer Vision - Predict Masked and Clean Identities

Predict labels of images using specified model, directory of clean images and
directory of masked images and compare the accuracy.

Example usage:
    python predict_compare.py --test-dir test_subset --mask-dir masked --model models/model_pretrained_vgg --output-file compare_predictions.csv 

@author: K. Nolle
"""

import argparse
import cv2
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf

import helper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', help='Path of directory clean images to predict labels of', type=str)
    parser.add_argument('--mask-dir', help='Path of directory masked images to predict labels of', type=str)
    parser.add_argument('--model', help='Name of the model', type=str)
    parser.add_argument('--output-file', help='(Optional) Name of the file to store results in', type=str)
    
    args = parser.parse_args()
    
    if args.test_dir is None:
        print("Please specify the directory the images to predict the labels of")
        exit(1)
        
    if args.mask_dir is None:
        print("Please specify the directory the images to predict the labels of")
        exit(1)

    if args.model is None:
        print("Please specify the name of the model")
        exit(1)
        
        
    # Load model architecture
    with open(args.model+".json", "r") as file:
        model_json = file.read()
        
    model = tf.keras.models.model_from_json(model_json) 
    
    # Load trained weights
    model.load_weights(args.model+'.h5')
    
    
    df_results = pd.DataFrame(columns=["filename_clean", "filename_mask", "target_label", "predicted_label_clean", "predicted_label_mask"])
    
    clean_file_ext = ".jpg"
    
    for filename_mask in os.listdir(args.mask_dir):
        print(filename_mask)
        tmp = filename_mask.split('-')
        
        label = tmp[1] # Get ID nr.
        filename_clean = tmp[0]+'-'+tmp[1]+clean_file_ext
        
        # Predict masked image
        # --------------------
        
        # Read image
        raw_data = cv2.imread(os.path.join(args.mask_dir, filename_mask))
        raw_data = cv2.resize(raw_data, (helper.WIDTH, helper.HEIGHT))
        
        # Scale input pixels to the range [0, 1] for Keras
        X = np.array(raw_data) / 255.0
        
        # Reshape to fit model input shape
        (h, w, c) = X.shape
        X = X.reshape([-1, h, w, c])
        
        # Predict labels
        y_prob = model.predict(X)
        
        # Select label with the highest probability
        prediction_mask = np.argmax(np.array(y_prob[0]))
        
        
        # Predict clean image
        # -------------------
        
        # Read image
        raw_data = cv2.imread(os.path.join(args.test_dir, filename_clean))
        raw_data = cv2.resize(raw_data, (helper.WIDTH, helper.HEIGHT))
        
        # Scale input pixels to the range [0, 1] for Keras
        X = np.array(raw_data) / 255.0
        
        # Reshape to fit model input shape
        (h, w, c) = X.shape
        X = X.reshape([-1, h, w, c])
        
        # Predict labels
        y_prob = model.predict(X)
        
        # Select label with the highest probability
        prediction_clean = np.argmax(np.array(y_prob[0]))
        
        df_results.loc[len(df_results)] = [filename_clean, filename_mask, label, prediction_clean, prediction_mask]
        
        
    # Cast label columns to int so that they can be compared
    df_results["target_label"] = df_results["target_label"].astype("string")
    df_results["predicted_label_clean"] = df_results["predicted_label_clean"].astype("string")
    df_results["predicted_label_mask"] = df_results["predicted_label_mask"].astype("string")
    
    # Calculate and print accuracy per class and overall
    correct = len(df_results[df_results["target_label"] == df_results["predicted_label_clean"]])
    total = len(df_results)
    print(f"Correct (clean): {correct}")
    print(f"Total (clean): {total}")
    print(f"Accuracy (clean): {correct/total}")
    
    print("\n\n")
    
    # Calculate and print accuracy per class and overall
    correct = len(df_results[df_results["target_label"] == df_results["predicted_label_mask"]])
    total = len(df_results)
    print(f"Correct (mask): {correct}")
    print(f"Total (mask): {total}")
    print(f"Accuracy (mask): {correct/total}")
    
    # Save results
    if args.output_file is not None:
        print(f"Saving results to {args.output_file}")
        df_results.to_csv(args.output_file)
    
    
if __name__ == '__main__':
    main()