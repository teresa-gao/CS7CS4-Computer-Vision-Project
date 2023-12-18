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
WIDTH = 178
HEIGHT = 218


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


RESNET_18_BLOCKS = [{'name' : 'conv2_',
                     'repeat' : 2,
                     'blocks' : [{'shape' : (3, 3),
                                  'filters' : 64},
                                 {'shape' : (3, 3),
                                  'filters' : 64},]
                     },
                    {'name' : 'conv3_',
                     'repeat' : 2,
                     'blocks' : [{'shape' : (3, 3),
                                  'filters' : 128},
                                 {'shape' : (3, 3),
                                  'filters' : 128},]
                     },
                    {'name' : 'conv4_',
                     'repeat' : 2,
                     'blocks' : [{'shape' : (3, 3),
                                  'filters' : 256},
                                 {'shape' : (3, 3),
                                  'filters' : 256},]
                     },
                    {'name' : 'conv5_',
                     'repeat' : 2,
                     'blocks' : [{'shape' : (3, 3),
                                  'filters' : 256},
                                 {'shape' : (3, 3),
                                  'filters' : 256},]
                     },
                   ]


# Build a Keras model given some parameters
def create_model(resnet_blocks, n_labels=LABELS, input_shape=(HEIGHT, WIDTH, 3)):

    input_tensor = keras.Input(input_shape) 
    x = input_tensor
  
    # conv1
    x = keras.layers.Conv2D(64, (7,7), padding='same', input_shape=input_shape, name="conv1")(x)
    x = keras.layers.BatchNormalization(name="conv1_norm")(x)
    x = keras.layers.Activation('relu', name="conv1_activate")(x)
  
    # max-pooling
    x = keras.layers.MaxPooling2D((3, 3), strides=2)(x)
  
    # first block
    for block in resnet_blocks:
        for i in range (block["repeat"]):
            x_skip = x
           
            j = 0
            for layer in block["blocks"]:
                if block["name"] in ['conv3_', 'conv4_', 'conv5_'] and i == 0 and j == 0:
                    strides = (2, 2)
                else:
                    strides = (1, 1)
                    
                x = keras.layers.Conv2D(layer["filters"], layer["shape"], strides=strides, padding='same', name=block["name"]+str(i)+str(j))(x)
                x = keras.layers.BatchNormalization(name=block["name"]+str(i)+str(j)+"_norm")(x)
                x = keras.layers.Activation('relu', name=block["name"]+str(i)+str(j)+"_activate")(x)
              
                j += 1
              
            if x_skip.shape != x.shape:
                if block["name"] in ['conv3_', 'conv4_', 'conv5_'] and i == 0:
                    strides = (2, 2)
                else:
                    strides = (1, 1)
                
                x_skip = keras.layers.Conv2D(x.shape[-1], (1,1), strides=strides, padding='same')(x_skip)
                
            x = keras.layers.Add()([x, x_skip])

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(n_labels, activation='softmax')(x)
    model = keras.Model(inputs=input_tensor, outputs=x)

    return model

