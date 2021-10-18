#!/usr/bin/python3

import tensorflow as tf;

def Head():
  inputs = tf.keras.Input((None, None, None, 64)); # inputs.shape = (batch, length, height, width, 64)
  reshaped = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[2], tf.shape(x)[3], 64)))(inputs); # reshaped.shape = (batch * length, height, width, 64)
  results = tf.keras.layers.BatchNormalization()(reshaped);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(64, kernel_size = (1,1), padding = 'valid')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(16, kernel_size = (3,3), padding = 'same')(results);
  block1_results = tf.keras.layers.Concatenate(axis = -1)([reshaped, results]); # block1_results.shape = (batch * length, height, width, 64 + 16)
  results = tf.keras.layers.BatchNormalization()(block1_results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(64 + 16, kernel_size = (1,1), padding = 'valid')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(16, kernel_size = (3,3), padding = 'same')(results);
  block2_results = tf.keras.layers.Concatenate(axis = -1)([block1_results, results]); # block2_results.shape = (batch * length, height, width, 64 + 16 + 16)
  results = tf.keras.layers.BatchNormalization()(block2_results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(64 + 16 + 16, kernel_size = (1,1), padding = 'valid')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(16, kernel_size = (3,3), padding = 'same')(results);
  block3_results = tf.keras.layers.Concatenate(axis = -1)([block2_results, results]); # block3_results.shape = (batch * length, height, width, 64 + 16 + 16 + 16)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (-1, tf.shape(x[1])[1], tf.shape(x[1])[2], tf.shape(x[1])[3], x[0].shape[-1])))([block3_results, inputs])
  return tf.keras.Model(inputs = inputs, outputs = results);

def Middle(channels):
  inputs = tf.keras.Input((None, None, None, channels)); # inputs.shape = (batch, length, height, width, channels)
  results = 

def TGA():
  inputs = tf.keras.Input((None, None, None, 3)); # inputs.shape = (batch, length, height, width, channels)
  reshaped = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[-3], tf.shape(x)[-2], x.shape[-1])))(inputs); # reshaped.shape = (batch * length, height, width, channels)
  # 1) Intra-group Fusion
  results = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same')(reshaped);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  # 2) temporal feature fusion
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (-1, tf.shape(x[1])[1], tf.shape(x[1])[2], tf.shape(x[1])[3], x[0].shape[-1])))([results, inputs]); # results.shape = (batch, length, height, width, 64)
  results = tf.keras.layers.Conv3D(64, kernel_size = (3,3,3), strides = (3,1,1), padding = 'same')(results); # results.shape = (batch, length, height, width, 64)
  results = Head()(results); # results.shape = (batch, length, height, width, 64 + 16 * 3)

  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":
  tga = TGA();
  import numpy as np;
  inputs = np.random.normal(size = (4,5,10,10,3));
  outputs = tga(inputs);
  print(inputs.shape);
  print(outputs.shape);

