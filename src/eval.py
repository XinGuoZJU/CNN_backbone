import os
import configparser
import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf

from utils.datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.data import Iterator
from tensorflow.keras.layers import Input

"""
Configuration Part.
"""

task_name = sys.argv[1]
config = configparser.ConfigParser()
config.read(os.path.join('configs', task_name+'.ini'))

# Path to the textfiles for the trainings and validation set
task_type = config.get('model', 'task')
mode = config.get('model', 'mode')  
sub_mode = config.getint('model', 'sub_mode') 
lower_bound = config.getfloat('model', 'lower_bound')
upper_bound = config.getfloat('model', 'upper_bound')
interval = config.getfloat('model', 'interval')

if task_type == 'reg':
    label_params = None
    num_classes = 1
elif task_type == 'cls':
    label_params = [lower_bound, interval]
    num_classes = int((upper_bound - lower_bound) / interval) + 1
else:
    raise ValueError("Invalid task_type '%s'." % (task_type))

image_path = config.get('data', 'image_path')
val_file = config.get('data', 'test_file')
checkpoint_path = 'logs/'+task_name+'/checkpoints'

# Learning params
batch_size = config.getint('train', 'batch_size')
threshold_num = config.getint('eval', 'threshold_num')

# Network params
if mode == 'alexnet':
    img_size = [227, 227]
elif mode == 'densenet':
    img_size = [224, 224]
elif mode == 'resnet':
    img_size = [224, 224]
else:
    raise ValueError("Invalid mode '%s'." % (mode))


"""
Main Part of the finetuning Script.
"""
# Create parent path if it doesn't exist
# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    val_data = ImageDataGenerator(image_path,
                                img_size,
                                val_file,
                                label_params,
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
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# Start Tensorflow session
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
    saver.restore(sess,ckpt.model_checkpoint_path)

    # Load the pretrained weights into the non-trainable layer
    graph = tf.get_default_graph()
    
    # Evaluation op: Accuracy of the model
    if mode == 'alexnet' or mode == 'resnet':
        x = graph.get_tensor_by_name('Placeholder:0')
        y = graph.get_tensor_by_name('Placeholder_1:0')
        keep_prob = graph.get_tensor_by_name('Placeholder_2:0')
        if mode == 'alexnet':
            score = graph.get_tensor_by_name('fc8-f/fc8-f:0')
        else:
            score = graph.get_tensor_by_name('resnet_model/final_dense:0')
    elif mode == 'densenet':
        x = graph.get_tensor_by_name('data:0')
        y = graph.get_tensor_by_name('Placeholder:0')
        keep_prob = graph.get_tensor_by_name('Placeholder_1:0')
        score = graph.get_tensor_by_name('fc6-f/BiasAdd:0')   

    threshold_inter = (upper_bound - lower_bound) / (threshold_num + 2) #???
    with tf.name_scope("accuracy"):
        accuracy = []
        if task_type == 'reg':
            vec_distance = tf.squeeze(tf.abs(score - y))
        elif task_type == 'cls':
            vec_distance = tf.math.scalar_mul(interval, 
                            tf.abs(tf.cast(tf.argmax(score, 1) - tf.argmax(y, 1), tf.float32)))
            
        one = tf.ones_like(vec_distance)
        zero = tf.zeros_like(vec_distance)
        
        for inter_num in range(threshold_num):
            threshold = lower_bound + threshold_inter * (inter_num + 1)
            idx = tf.where(vec_distance < threshold, x=one, y=zero)
            accuracy.append(tf.count_nonzero(idx)/batch_size)

    # Validate the model on the entire validation set
    print("{} Start validation".format(datetime.now()))
    sess.run(validation_init_op)
    test_acc = [0 for _ in range(threshold_num)]
    test_count = 0
    for iter_epoch in range(val_batches_per_epoch):
        print("Epoch {} of {}".format(iter_epoch, val_batches_per_epoch))

        img_batch, label_batch = sess.run(next_batch)
        acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})

        test_acc = [test_acc[i]+acc[i] for i in range(threshold_num)]
        test_count += 1
    
    test_acc = [item/test_count for item in test_acc]  
    print(test_acc)

    x_axis = np.asarray([(lower_bound + threshold_inter * ( i + 1)) / np.pi * 180 for i in range(threshold_num)])
    y_axis = np.asarray(test_acc)
    result_path = 'results'
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    np.savez(os.path.join(result_path, task_name + '.npz'), x_f = x_axis, y_f = y_axis)
    
    '''
    plt.xlabel('FoV Error Threshold (deg)')
    plt.ylabel('Correct Rate')
    plt.plot(x_axis, y_axis, label='AUC')
    plt.legend()
    plt.savefig('result.png')
    '''

