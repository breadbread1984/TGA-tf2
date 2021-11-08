#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
from absl import flags, app;
import tensorflow as tf;
from models import TGA;
from create_dataset import Vimeo90k;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_integer('batch_size', default = 16, help = 'batch size');
  flags.DEFINE_string('vimeo_path', default = None, help = 'dataset path');
  flags.DEFINE_boolean('save_model', default = False, help = 'whether to save the trained model');
  flags.DEFINE_integer('checkpoint_steps', default = 1000, help = 'how many steps for each checkpoint');
  flags.DEFINE_integer('eval_steps', default = 1000, help = 'how many steps for each evaluation');
  flags.DEFINE_integer('epochs', default = 560, help = 'epoch number');
  flags.DEFINE_float('lr', default = 1e-3, help = 'learning rate');

class SummaryCallback(tf.keras.callbacks.Callback):
  def __init__(self, tga, eval_freq = 1000):
    self.tga = tga;
    self.eval_freq = eval_freq;
    testset = Vimeo90k(FLAGS.vimeo_path).load_datasets().batch(1).repeat(-1);
    self.iter = iter(testset);
    self.log = tf.summary.create_file_writer('checkpoints');
  def on_batch_end(self, batch, logs = None):
    if batch % self.eval_freq == 0:
      lr, hr = next(self.iter);
      pred_hr = tf.cast(self.tga(lr) * 255., dtype = tf.uint8); # pred_hr.shape = (1, h, w ,3)
      gt_hr = tf.cast(hr['hr'] * 255., dtype = tf.uint8); # gt_hr.shape = (1, h, w, 3)
      with self.log.as_default():
        for key, value in logs.items():
          tf.summary.scalar(key, value, step = self.tga.optimizer.iterations);
        tf.summary.image('ground truth', gt_hr, step = self.tga.optimizer.iterations);
        tf.summary.image('predict', pred_hr, step = self.tga.optimizer.iterations);

def train():
  if exists(join('checkpoints', 'ckpt')):
    tga = tf.keras.models.load_model(join('checkpoints', 'ckpt'), custom_objects = {'tf': tf}, compile = True);
    optimizer = tga.optimizer;
  else:
    tga = TGA();
    optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(FLAGS.lr, decay_steps = 100));
    tga.compile(optimizer = optimizer,
                loss = {'hr': tf.keras.losses.MeanAbsoluteError()},
                metrics = [tf.keras.metrics.MeanAbsoluteError()]);
  trainset = Vimeo90k(FLAGS.vimeo_path).load_datasets().batch(FLAGS.batch_size);
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = 'checkpoints'),
    tf.keras.callbacks.ModelCheckpoint(filepath = join('checkpoints', 'ckpt'), save_freq = FLAGS.checkpoint_steps),
    SummaryCallback(tga, FLAGS.eval_steps),
  ];
  tga.fit(trainset, epochs = FLAGS.epochs, callbacks = callbacks);
  tga.save_weights('tga_weights.h5');

def save_model():
  if not exists('models'): mkdir('models');
  tga = tf.keras.models.load_model(join('checkpoints', 'ckpt'), custom_objects = {'tf': tf}, compile = True);
  tga.save(join('models', 'tga.h5'));
  tga.save_weights(join('models', 'tga_weights.h5'));

def main(unused_argv):
  if FLAGS.save_model:
    save_model();
  else:
    train();

if __name__ == "__main__":
  add_options();
  app.run(main);
