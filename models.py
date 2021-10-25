#!/usr/bin/python3

import tensorflow as tf;

def Unit(in_channels = 64, out_channels = 16, is_last = False):
  inputs = tf.keras.Input((None, None, None, in_channels)); # inputs.shape = (batch, length, height, width, 64)
  reshaped = tf.keras.layers.Lambda(lambda x, c: tf.reshape(x, (-1, tf.shape(x)[2], tf.shape(x)[3], c)), arguments = {'c': in_channels})(inputs); # reshaped.shape = (batch * length, height, width, 64)
  results = tf.keras.layers.BatchNormalization()(reshaped);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(in_channels, kernel_size = (1,1), padding = 'valid')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(out_channels, kernel_size = (3,3), padding = 'same')(results);
  block1_results = tf.keras.layers.Concatenate(axis = -1)([reshaped, results]); # block1_results.shape = (batch * length, height, width, 64 + 16)
  results = tf.keras.layers.BatchNormalization()(block1_results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(in_channels + out_channels, kernel_size = (1,1), padding = 'valid')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(out_channels, kernel_size = (3,3), padding = 'same')(results);
  block2_results = tf.keras.layers.Concatenate(axis = -1)([block1_results, results]); # block2_results.shape = (batch * length, height, width, 64 + 16 + 16)
  results = tf.keras.layers.BatchNormalization()(block2_results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(in_channels + out_channels * 2, kernel_size = (1,1), padding = 'valid')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  if is_last == True:
    results = tf.keras.layers.Conv2D(out_channels + 1, kernel_size = (3,3), padding = 'same')(results); # results.shape = (batch * length, height, width, 17)
    attention = tf.keras.layers.Lambda(lambda x: x[...,-1:])(results); # attention.shape = (batch * length, height, width, 1)
    attention = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (-1, tf.shape(x[1])[1], tf.shape(x[0])[1], tf.shape(x[0])[2], 1)))([attention, inputs]); # attention.shape = (batch, length, height, width, 1)
    attention = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x, axis = 1))(attention); # attention.shape = (batch, length, height, width, 1)
    attention = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[-3], tf.shape(x)[-2], 1)))(attention); # attention.shape = (batch * length, height, width, 1)
    features = tf.keras.layers.Lambda(lambda x: x[...,:-1])(results); # features.shape = (batch * length, height, width, 16)
    results = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([attention, features]); # results.shape = (batch * length, height, width, 16)
  else:
    results = tf.keras.layers.Conv2D(out_channels, kernel_size = (3,3), padding = 'same')(results); # results.shape = (batch * length, height, width, 16)
  block3_results = tf.keras.layers.Concatenate(axis = -1)([block2_results, results]); # block3_results.shape = (batch * length, height, width, 64 + 16 + 16 + 16)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (-1, tf.shape(x[1])[1], tf.shape(x[1])[2], tf.shape(x[1])[3], x[0].shape[-1])))([block3_results, inputs])
  return tf.keras.Model(inputs = inputs, outputs = results);

def Fusion(in_channels, out_channels, is_last = False):
  inputs = tf.keras.Input((None, None, None, in_channels)); # inputs.shape = (batch, length, height, width, channels)
  results = inputs;
  if is_last == True:
    results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv3D(in_channels, kernel_size = (1,1,1), padding = 'valid')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv3D(out_channels, kernel_size = (3,3,3), padding = 'same')(results);
  if is_last == True:
    # NOTE: the following layer makes length = length - 2
    results = tf.keras.layers.Lambda(lambda x: x[:,1:-1,...])(results);
    sliced_inputs = tf.keras.layers.Lambda(lambda x: x[:,1:-1,...])(inputs); # results.shape = (batch, length - 1, height, width, channels)
  else:
    sliced_inputs = inputs; # results.shape = (batch, length, height, width, channels)
  block1_results = tf.keras.layers.Concatenate(axis = -1)([sliced_inputs, results]);
  results = tf.keras.layers.BatchNormalization()(block1_results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv3D(in_channels + out_channels, kernel_size = (1,1,1), padding = 'valid')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  if is_last == True:
    results = tf.keras.layers.Conv3D(out_channels, kernel_size = (1,3,3), padding = 'same')(results);
  else:
    results = tf.keras.layers.Conv3D(out_channels, kernel_size = (3,3,3), padding = 'same')(results);
  block2_results = tf.keras.layers.Concatenate(axis = -1)([block1_results, results]);
  return tf.keras.Model(inputs = inputs, outputs = block2_results);

def TGA():
  inputs = tf.keras.Input((7, None, None, 3)); # inputs.shape = (batch, 7 frames in temporal sequence, height, width, channels)
  reordered = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.gather(tf.transpose(x, (1,0,2,3,4)), [0,3,6,2,3,4,1,3,5]), (1,0,2,3,4)))(inputs); # results.shape = (batch, 9, height, width, channels)
  reshaped = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[-3], tf.shape(x)[-2], x.shape[-1])))(reordered); # reshaped.shape = (batch * length, height, width, channels)
  # 1) Intra-group Fusion
  # first part (2d unit x 3)
  results = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same')(reshaped);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  # second part (3d unit x 1)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (-1, tf.shape(x[1])[1], tf.shape(x[1])[2], tf.shape(x[1])[3], x[0].shape[-1])))([results, reordered]); # results.shape = (batch, length, height, width, 64)
  # NOTE: the following layer make length = length / 3
  results = tf.keras.layers.Conv3D(64, kernel_size = (3,3,3), strides = (3,1,1), padding = 'same')(results); # results.shape = (batch, length, height, width, 64)
  # 2) Inter-group Fusion
  results = Unit(64, 16)(results); # results.shape = (batch, length, height, width, 64 + 16 * 3)
  results = Unit(64 + 16 * 3, 16)(results);
  results = Unit(64 + 16 * 6, 16)(results);
  results = Unit(64 + 16 * 9, 16)(results);
  results = Unit(64 + 16 * 12, 16)(results);
  results = Unit(64 + 16 * 15, 16, is_last = True)(results); # attention happends at this layer
  results = Fusion(64 + 16 * 18, 16)(results);
  # NOTE: the following layer make length = length - 2
  results = Fusion(64 + 16 * 20, 16, is_last = True)(results);
  reshaped = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[2], tf.shape(x)[3], x.shape[4])))(results);
  reshaped = tf.keras.layers.BatchNormalization()(reshaped);
  reshaped = tf.keras.layers.ReLU()(reshaped);
  reshaped = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same')(reshaped);
  results = results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (-1, tf.shape(x[1])[1], tf.shape(x[1])[2], tf.shape(x[1])[3], x[0].shape[-1])))([reshaped, results]);
  # 3) sub-pixel upsampling x4 to get laplacian
  results = Unit(64, 16)(results);
  results = Unit(64 + 16 * 3, 16)(results);
  results = Unit(64 + 16 * 6, 16)(results);
  results = Unit(64 + 16 * 9, 16)(results);
  results = Unit(64 + 16 * 12, 16)(results);
  results = Unit(64 + 16 * 15, 16)(results);
  results = Unit(64 + 16 * 18, 16)(results);
  reshaped = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[2], tf.shape(x)[3], x.shape[4])))(results);
  results = tf.keras.layers.BatchNormalization()(reshaped);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(256, kernel_size = (3,3), padding = 'same')(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(256, kernel_size = (1,1), padding = 'same')(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(128 * 4, kernel_size = (3,3), padding = 'same')(results); # results = (batch * length = 1, height, width, channels)
  results = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(results); # results.shape = (batch, height * 2, width * 2, channels / 4)
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(128, kernel_size = (1,1), padding = 'same', activation = tf.keras.activations.relu)(results);
  results = tf.keras.layers.Conv2D(3 * 4, kernel_size = (1,1), padding = 'same')(results);
  laplacian = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(results); # results.shape = (batch, height * 2, width * 2, channels / 4)
  # 4) bicubic upsampling the reference frames
  references = tf.keras.layers.Lambda(lambda x: x[:, 3, ...])(inputs); # references.shape = (batch, height, width, channels)
  gaussian = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (tf.shape(x)[1] * 4, tf.shape(x)[2] * 4), method = tf.image.ResizeMethod.BICUBIC))(references);
  # 5) combine to get the final super resolution images
  results = tf.keras.layers.Add(name = 'hr')([laplacian, gaussian]);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":
  tga = TGA();
  import numpy as np;
  inputs = np.random.normal(size = (4,7,10,10,3));
  outputs = tga(inputs);
  print(inputs.shape);
  print(outputs.shape);
