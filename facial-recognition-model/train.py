# -*- coding: utf-8 -*-
"""
CS7GV1 Computer Vision - ResNet Model Training

This script trains ResNet models on data from a specified directory

Example usage:
    python train.py --train-dir train --val-dir validation --batch-size 32 --output-model-name test --epochs 5

@author: K. Nolle
"""

import argparse
import cv2
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.data import Dataset
import tensorflow.keras as keras

#from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace

import h5py

import helper
    
# Set environment variables to avoid OOM errors
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def create_model(n_labels, input_shape, hidden_dim=512):
    vgg_model = VGGFace(include_top=False, input_shape=input_shape)#, model='resnet50')
    vgg_model.trainable = False # Freeze base model
    last_layer = vgg_model.get_layer('pool5').output
    
    # Convolution layer to account for different image scales
    # (Images are downsampled in pre-processing)
    #x = keras.layers.Conv2D(16, (3,3), padding='same', input_shape=input_shape,activation='relu')(last_layer)
    #x = keras.layers.BatchNormalization()(x)
    
    # Fully connected layers for classification
    x = Flatten(name='flatten')(last_layer)#(x)
    x = Dense(hidden_dim, activation='relu', name='fc6')(x)
    x = Dense(hidden_dim, activation='relu', name='fc7')(x)
    out = Dense(n_labels, activation='softmax', name='fc8')(x)

    return keras.Model(vgg_model.input, out)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', help='Path of directory with training data', type=str)
    parser.add_argument('--val-dir', help='Path of directory with validation data', type=str)
    parser.add_argument('--batch-size', help='Batch size for training', type=int)
    parser.add_argument('--output-model-name', help='Where to save the trained model', type=str)
    parser.add_argument('--input-model', help='Where to look for the input model to continue training', type=str)
    parser.add_argument('--epochs', help='Maximum number of epochs to run (note: early-stopping is used)', type=int)
    
    args = parser.parse_args()

    if args.train_dir is None:
        print("Please specify the directory with training data")
        exit(1)
        
    if args.val_dir is None:
        print("Please specify the directory with validation data")
        exit(1)

    if args.batch_size is None:
        print("Please specify the training batch size")
        exit(1)

    if args.epochs is None:
        print("Please specify the number of training epochs to run")
        exit(1)


    if args.output_model_name is None:
        print("Please specify a name for the trained model")
        exit(1)


    # Check if GPU is available
    #if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    #    device = 'GPU:0'
    #else:
    #    device = '/device:CPU:0'
    
    # Clear any previous states
    tf.keras.backend.clear_session()
    
    device = '/device:CPU:0'

    with tf.device(device):
        #model = helper.create_model(helper.RESNET_18_BLOCKS)
        
        model = create_model(helper.LABELS, (helper.HEIGHT, helper.WIDTH, 3))
        
        if args.input_model is not None:
            model.load_weights(args.input_model)

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-4, amsgrad=True),
                      metrics=['accuracy'])

        model.summary()
        
        callbacks = [keras.callbacks.EarlyStopping(patience=3, verbose=1),
                     keras.callbacks.CSVLogger(f"log_{args.output_model_name}.csv"),
                     keras.callbacks.ModelCheckpoint(args.output_model_name+'.h5', save_best_only=True),
                     #keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-4)
                     ]

        # Save the model architecture to JSON
        with open(args.output_model_name+".json", "w") as json_file:
            json_file.write(model.to_json())

        try:
            model.fit(helper.create_dataset(args.train_dir, args.batch_size),
                      validation_data=helper.create_dataset(args.val_dir, args.batch_size),
                      epochs=args.epochs,
                      callbacks=callbacks,
                      use_multiprocessing=True
                      )
            
        except KeyboardInterrupt:
            print('KeyboardInterrupt caught, saving current weights as ' + args.output_model_name+'_resume.h5')
            model.save_weights(args.output_model_name+'_resume.h5')
        except  Exception as e:
            raise e
            

if __name__ == '__main__':
    main()