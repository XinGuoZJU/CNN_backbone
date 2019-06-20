import os
import configparser
import math
import shutil
import sys
import numpy as np
import tensorflow as tf

from model.alexnet import AlexNet
from model.densenet import DenseNet
from model.resnet import ResNet
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

if task_type == 'reg':
    label_params = None
    num_classes = 1
elif task_type == 'cls':
    lower_bound = config.getfloat('model', 'lower_bound')
    upper_bound = config.getfloat('model', 'upper_bound')
    interval = config.getfloat('model', 'interval')
    label_params = [lower_bound, interval]
    num_classes = int((upper_bound - lower_bound) / interval) + 1
else:
    raise ValueError("Invalid task_type '%s'." % (task_type)) 

image_path = config.get('data', 'image_path')
train_file = config.get('data', 'train_file')
val_file = config.get('data', 'val_file')

# Learning params
learning_rate = config.getfloat('train', 'learning_rate')
decay_steps = config.getint('train', 'decay_steps')
decay_rate = config.getfloat('train', 'decay_rate')
num_epochs = config.getint('train', 'num_epochs')
batch_size = config.getint('train', 'batch_size')
train_layers = eval(config.get('train','train_layers'))
pretrain = config.getboolean('train', 'pretrain')

# Network params
if mode == 'alexnet':
    img_size = [227, 227]
elif mode == 'densenet':
    img_size = [224, 224]
elif mode == 'resnet':
    img_size = [224, 224]
else:
    raise ValueError("Invalid mode '%s'." % (mode))

dropout_rate = 0.5

# How often we want to write the tf.summary data to disk
display_step = 20
val_step = 5

# Path for tf.summary.FileWriter and to store model checkpoints
log_path = os.path.join('logs', task_name)
filewriter_path = os.path.join(log_path, 'tensorboard')
checkpoint_path = os.path.join(log_path, 'checkpoints')

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
shutil.copy(os.path.join('configs', task_name+'.ini'), log_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(image_path,
                                img_size,
                                train_file,
                                label_params, 
                                mode='training',
                                batch_size=batch_size,
                                num_classes=num_classes,
                                shuffle=True)
    val_data = ImageDataGenerator(image_path,
                                img_size,
                                val_file,
                                label_params,
                                mode='inference',
                                batch_size=batch_size,
                                num_classes=num_classes,
                                shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
if mode == 'alexnet' or mode == 'resnet':
    x = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 3])
elif mode == 'densenet':
    x = Input(shape=(img_size[0], img_size[1], 3), name='data')
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False)

# Initialize model
if mode == 'alexnet':
    model = AlexNet(x, keep_prob, num_classes, train_layers)
    score = model.fc8
elif mode == 'densenet':
    model_op = DenseNet(sub_mode, x, num_classes=num_classes)
    model = model_op.create()
    score = model_op.output
elif mode == 'resnet':
    model_op = ResNet(resnet_size=sub_mode, num_classes=num_classes, resnet_version=1) 
    score = model_op.create(x, True)

# List of trainable variables of the layers we want to train
if 'all' in train_layers:
    var_list = tf.trainable_variables()
else:
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))
    if task_type == 'reg':
        loss = tf.losses.mean_squared_error(predictions=score, labels=y)
    elif task_type == 'cls':
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    learning_rate_tensor = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase = True)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate_tensor)
    optimizer = tf.train.AdamOptimizer(learning_rate_tensor)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

# Add gradients to summary
# for gradient, var in gradients:
#     tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
# for var in var_list:
#     tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    # correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    if task_type == 'reg':
        accuracy = tf.reduce_sum(tf.square(score - y))
    elif task_type == 'cls':
        correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1)) 
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver(max_to_keep=10)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        pass
    else:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


    # Load the pretrained weights into the non-trainable layer
    if pretrain and mode == 'alexnet':
        model.load_initial_weights(sess)
    elif pretrain and mode == 'densenet':
        model_op.load_weights()
    elif pretrain and mode == 'resnet':
        raise ValueError("Invalid pretrain for resnet") 
    
    if ckpt and ckpt.model_checkpoint_path:
        print('Training from saved model ...')
        saver.restore(sess, ckpt.model_checkpoint_path)
        graph = tf.get_default_graph()
    
    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    # for epoch in range(num_epochs):

    epoch = int(global_step.eval()/train_batches_per_epoch)
    while(epoch < num_epochs):
        epoch += 1

        print("{} Epoch number: {}".format(datetime.now(), epoch))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, (epoch-1)*train_batches_per_epoch + step)

        if epoch % val_step == 0:
            # Validate the model on the entire validation set
            print("{} Start validation".format(datetime.now()))
            sess.run(validation_init_op)
            test_acc = 0.
            test_count = 0
            for _ in range(val_batches_per_epoch):

                img_batch, label_batch = sess.run(next_batch)
                acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
                test_acc += acc
                test_count += 1
            test_acc /= test_count
            print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
