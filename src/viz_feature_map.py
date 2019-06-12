import os
import cv2
import configparser
import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf

from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.data import Iterator

"""
Configuration Part.
"""

task_name = sys.argv[1]
config = configparser.ConfigParser()
config.read(os.path.join('configs', task_name+'.ini'))

# Path to the textfiles for the trainings and validation set
image_path = config.get('train', 'image_path')
val_file = config.get('eval', 'test_file')
checkpoint_path = 'logs/'+task_name+'/checkpoints'

# Learning params
batch_size = config.getint('train', 'batch_size')
# batch_size = 1
lower_bound = config.getfloat('eval', 'lower_bound')
upper_bound = config.getfloat('eval', 'upper_bound')
interval = config.getfloat('eval', 'interval')
threshold_num = config.getint('eval', 'threshold_num')

# Network params
img_size = [227, 227]
label_params = [lower_bound, interval]
num_classes = int((upper_bound - lower_bound) / interval) + 1


"""
Main Part of the finetuning Script.
"""
# Create parent path if it doesn't exist
# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    val_data = ImageDataGenerator(image_path,
                                  img_size,
                                  val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(val_data.data.output_types,
                                       val_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
validation_init_op = iterator.make_initializer(val_data.data)

# Get the number of training/validation steps per epoch
# val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))
val_batches_per_epoch = 1 

fm_path = os.path.join('feature_map', task_name)
if not os.path.isdir(fm_path):
    os.makedirs(fm_path)

# Start Tensorflow session
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
    saver.restore(sess,ckpt.model_checkpoint_path)

    # Load the pretrained weights into the non-trainable layer
    graph = tf.get_default_graph()
    
    # Evaluation op: Accuracy of the model
    x = graph.get_tensor_by_name('Placeholder:0')
    # x = tf.placeholder(tf.float32, [batch_size, img_size[0], img_size[1], 3])
    # y = graph.get_tensor_by_name('Placeholder_1:0')

    layer_list = ['conv5_1']
    img_idx = 0
    for layer_name in layer_list:
        feature_map = graph.get_tensor_by_name(layer_name+':0')
    
        print('Processing: {} of image {}'.format(layer_name, img_idx))
        sess.run(validation_init_op)
        for iter_epoch in range(val_batches_per_epoch):
            # print("Epoch {} of {}".format(iter_epoch, val_batches_per_epoch))

            img_batch, label_batch = sess.run(next_batch)
            fm = sess.run(feature_map, feed_dict={x: img_batch})
            
            fm_shape = fm.shape
            sum_suppress = 0
            for i in range(fm_shape[-1]):
                img = fm[img_idx, :, :, i]  # 32*13*13*256
                img_min = np.min(img)
                img_max = np.max(img)
                if img_min == img_max:
                    new_img = 0 * img
                    sum_suppress += 1
                    print('Single value image ! Value: {}'.format(img_min))
                else:
                    new_img = 255 * (img - img_min) / (img_max - img_min)

                cv2.imwrite(os.path.join(fm_path, '_'.join(layer_name.split('/'))+'-'+str(i)+'.jpg'), new_img)
            print('Number of Suppress: ', sum_suppress)
            print('Rate of Suppress: ', sum_suppress * 1.0/fm_shape[-1])



