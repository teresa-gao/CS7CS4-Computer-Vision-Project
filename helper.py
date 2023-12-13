# -*- coding: utf-8 -*-
"""
CS7GV1 Computer Vision - Helper

Library of functions and definitions

@author: K. Nolle
"""

#import argparse
import cv2
import numpy as np
import os
#import random
import tensorflow as tf
from tensorflow.data import Dataset
import tensorflow.keras as keras
#import h5py

LABELS = 500
WIDTH = 89
HEIGHT = 109


def create_dataset(directory_name, batch_size, img_height=HEIGHT, img_width=WIDTH, n_labels=LABELS):
    data = []
    target = []
    
    n_batches = int(np.floor(len(os.listdir(directory_name)) / batch_size))
    
    for filename in os.listdir(directory_name):
        label = filename.split('-')[-1] # Get ID nr.
        label = label.split('.')[0]     # Remove file extension
        
        # Read image
        raw_data = cv2.imread(os.path.join(directory_name, filename))
        raw_data = cv2.resize(raw_data, (img_width, img_height))
        
        # Scale input pixels to the range [0, 1] for Keras
        X = np.array(raw_data) / 255.0
            
        y = np.zeros(n_labels, dtype=np.uint8)
        y[int(label)] = 1
        
        data.append(X)
        target.append(y)
        
    # Create tensor on CPU (reduce memory needed on GPU)
    with tf.device("CPU"):
        ds = Dataset.from_tensor_slices((data, target)).shuffle(n_batches*batch_size).batch(batch_size)
        
    return ds



# Build a Keras model given some parameters
def create_model(n_labels=LABELS, input_shape=(HEIGHT, WIDTH, 3)):

  input_tensor = keras.Input(input_shape) 
  x = input_tensor
  x = keras.layers.Conv2D(16, (3,3), padding='same', input_shape=input_shape,activation='relu')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPooling2D(2)(x)
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dense(n_labels, activation='softmax')(x)
  model = keras.Model(inputs=input_tensor, outputs=x)

  return model

