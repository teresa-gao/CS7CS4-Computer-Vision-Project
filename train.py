# -*- coding: utf-8 -*-
"""
CS7GV1 Computer Vision - ResNet Model Training

This script trains ResNet models on data from a specified directory

Example usage:
    python train.py --batch-size 128 --output-model-name test --epochs 5

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

    
def create_dataset(directory_name, batch_size, img_height, img_width, n_labels):
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
def create_model(n_labels, input_shape):

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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', help='Batch size for training', type=int)
    parser.add_argument('--output-model-name', help='Where to save the trained model', type=str)
    parser.add_argument('--input-model', help='Where to look for the input model to continue training', type=str)
    parser.add_argument('--epochs', help='Maximum number of epochs to run (note: early-stopping is used)', type=int)
    
    args = parser.parse_args()

    if args.batch_size is None:
        print("Please specify the training batch size")
        exit(1)

    if args.epochs is None:
        print("Please specify the number of training epochs to run")
        exit(1)


    if args.output_model_name is None:
        print("Please specify a name for the trained model")
        exit(1)
        
    train_dir = "train"
    val_dir = "validation"
    labels = 500
    
    # To downsample image by 2
    img_width = 89
    img_height = 109


    # Check if GPU is available
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        device = 'GPU:0'
    else:
        device = '/device:CPU:0'

    with tf.device(device):
        model = create_model(labels, (img_height, img_width, 3))
        
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
            model.fit(create_dataset(train_dir, args.batch_size, img_height, img_width, labels),
                      validation_data=create_dataset(val_dir, args.batch_size, img_height, img_width, labels),
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