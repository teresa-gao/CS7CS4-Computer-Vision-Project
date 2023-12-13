# -*- coding: utf-8 -*-
"""
CS7GV1 Computer Vision - ResNet Model Training

This script trains ResNet models on data from a specified directory

Example usage:
    python train.py --train-dir train --val-dir validation --batch-size 128 --output-model-name test --epochs 5

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
import h5py

import helper
    

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
        
    #train_dir = "train"
    #val_dir = "validation"
    #labels = 500
    
    # To downsample image by 2
    #img_width = 89
    #img_height = 109


    # Check if GPU is available
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        device = 'GPU:0'
    else:
        device = '/device:CPU:0'

    with tf.device(device):
        model = helper.create_model()
        
        if args.input_model is not None:
            model.load_weights(args.input_model)

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-4, amsgrad=True),
                      metrics=['accuracy'])

        model.summary()
        
        callbacks = [keras.callbacks.EarlyStopping(patience=3, verbose=1),
                     keras.callbacks.CSVLogger('log.csv'),
                     keras.callbacks.ModelCheckpoint(args.output_model_name+'.h5', save_best_only=False)]

        # Save the model architecture to JSON
        with open(args.output_model_name+".json", "w") as json_file:
            json_file.write(model.to_json())

        try:
            model.fit(helper.create_dataset(args.train_dir, args.batch_size),
                      validation_data=helper.create_dataset(args.val_dir, args.batch_size),
                      epochs=args.epochs,
                      callbacks=callbacks,
                      use_multiprocessing=True)
            
        except KeyboardInterrupt:
            print('KeyboardInterrupt caught, saving current weights as ' + args.output_model_name+'_resume.h5')
            model.save_weights(args.output_model_name+'_resume.h5')
        except  Exception as e:
            raise e
            

if __name__ == '__main__':
    main()