#!/usr/bin/python3

from os.path import join;
from collections import deque;
from absl import flags, app;
import numpy as np;
import cv2;
import tensorflow as tf;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_string('model', default = join('models', 'tga.h5'), help = 'model path');
  flags.DEFINE_string('video', default = None, help = 'video path');
  flags.DEFINE_boolean('show', default = False, help = 'whether to show the super resolution video');

def main(unused_argv):
  tga = tf.keras.models.load_model(FLAGS.model, custom_objects = {'tf': tf});
  video = cv2.VideoCapture(FLAGS.video);
  fr = video.get(cv2.CAP_PROP_FPS);
  width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
  height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT));
  writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fr, (width * 4, height * 4));
  if video.isOpened() == False:
    print('invalid video');
    exit();
  img_buf = deque();
  retval = True;
  while retval:
    retval, img = video.read();
    if retval == False: break;
    img_buf.append(img);
    if len(img_buf) >= 7:
      while len(img_buf) > 7: img_buf.popleft();
      assert len(img_buf) == 7;
      inputs = np.expand_dims(np.array(img_buf), axis = 0).astype(np.float32) / 255.; # inputs.shape = (batch, 7, height, width, 3)
    else: continue;
    outputs = tga(inputs); # outputs.shape = (1, height, width, 3)
    sr = tf.squeeze(outputs * 255., axis = 0).numpy().astype(np.uint8);
    if FLAGS.show:
      cv2.imshow('sr', sr);
      cv2.waitKey(int(1e3 / fr));
    writer.write(sr);
  video.release();
  writer.release();

if __name__ == "__main__":
  add_options();
  app.run(main);
