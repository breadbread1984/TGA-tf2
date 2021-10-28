#!/usr/bin/python3

from os.path import join;
import numpy as np;
import cv2;
import tensorflow as tf;

class Vimeo90k(object):
  def __init__(self, root_dir):
    self.root_dir = root_dir;
    file_list = [line.rstrip() for line in open(join(root_dir, 'sep_trainlist.txt'))];
    self.image_set_list = [join(root_dir, 'sequences', x) for x in file_list];
  def generator(self):
    for image_set in self.image_set_list:
      # 1) load images
      hr = list();
      for i in range(7):
        img = cv2.imread(join(image_set, 'im%d.png' % (i + 1)))[...,::-1]; # convert from BGR to RGB
        h,w,_ = img.shape;
        h = h - (h % 4);
        w = w - (w % 4);
        img = img[:h,:w,:];
        hr.append(img);
      # 2) augmentation
      if np.random.uniform() < 0.5:
        hr = [cv2.flip(img, 1) for img in hr];
      if np.random.uniform() < 0.5:
        hr = [cv2.flip(img, 0) for img in hr];
      # 3) downsample
      lr = [cv2.resize(img, (w//4,w//4), interpolation = cv2.INTER_CUBIC) for img in hr];
      lr = np.array(lr); # lr.shape = (7, h/4, w/4, 3)
      yield lr.astype(np.float32), hr[3].astype(np.float32);
  def parse_function(self, lr, hr):
    lr = lr / 255.;
    hr = hr / 255.;
    return lr, {'hr': hr};
  def load_datasets(self):
    return tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.float32), (tf.TensorShape([7, None, None, 3]), tf.TensorShape([None, None, 3]))).map(self.parse_function);

if __name__ == "__main__":
  from sys import argv;
  if len(argv) != 2:
    print('Usage: %s <vimeo90k>');
    exit();
  dataset = Dataset(argv[1]);

