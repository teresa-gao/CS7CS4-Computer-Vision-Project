# -*- coding: utf-8 -*-
"""
CS7GV1 Computer Vision - Predict Identities

Predict labels of images using specified model and directory of images

@author: K. Nolle
"""

import argparse
import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf

import helper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='Path of directory images to predict labels of', type=str)
    parser.add_argument('--model', help='Name of the model', type=str)
    parser.add_argument('--output-file', help='(Optional) Name of the file to store results in', type=str)
    
    args = parser.parse_args()
    
    if args.dir is None:
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
    
    
    df_results = pd.DataFrame(columns=["filename", "target_label", "predicted_label"])
    
    for filename in os.listdir(args.dir):
        print(filename)
        label = filename.split('-')[-1] # Get ID nr.
        label = label.split('.')[0]     # Remove file extension
        
        # Read image
        raw_data = cv2.imread(os.path.join(args.dir, filename))
        raw_data = cv2.resize(raw_data, (helper.WIDTH, helper.HEIGHT))
        
        # Scale input pixels to the range [0, 1] for Keras
        X = np.array(raw_data) / 255.0
        
        # Reshape to fit model input shape
        (h, w, c) = X.shape
        X = X.reshape([-1, h, w, c])
        
        # Predict labels
        y_prob = model.predict(X)
        
        # Select label with the highest probability
        prediction = np.argmax(np.array(y_prob[0]))
        
        df_results.loc[len(df_results)] = [filename, label, prediction]
        
    # TODO: Calculate and print accuracy per class and overall
    
    
    # Save results
    if args.output_file is not None:
        print(f"Saving results to {args.output_file}")
        df_results.to_csv(args.output_file)
    
    
if __name__ == '__main__':
    main()