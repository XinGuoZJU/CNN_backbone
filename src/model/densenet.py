# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Concatenate, ZeroPadding2D
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

from sklearn.metrics import log_loss

from utils.scale_layer import Scale

# from load_cifar10 import load_cifar10_data

class DenseNet(object):
    def __init__(self, densenet_size, img_input, nb_dense_block=4, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, num_classes=None):
        '''
        DenseNet 121 Model for Keras

        Model Schema is based on 
        https://github.com/flyyufelix/DenseNet-Keras

        ImageNet Pretrained Weights 
        121:
        Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfMlRYb3YzV210VzQ
        TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfSTA4SHJVOHNuTXc
        161:
        Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfVnlCMlBGTDR3RGs 
        TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfUDZwVjU2cFNidTA
        169

        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
        '''
        self.densenet_size = densenet_size

        if self.densenet_size == 121:
            self.growth_rate = 32
            self.nb_filter = 64
            self.nb_layers = [6,12,24,16] # For DenseNet-121
        elif self.densenet_size == 161:
            self.growth_rate = 48
            self.nb_filter = 96
            self.nb_layers = [6,12,34,24] # For DenseNet-161
        elif self.densenet_size == 169:
            self.growth_rate = 32
            self.nb_filter = 64
            self.nb_layers = [6,12,32,32] # For DenseNet-161
        else:
            raise ValueError("Invalid mode '%s'." % (self.densenet_size))

        self.img_input = img_input
        self.img_rows = 224
        self.img_cols = 224
        self.color_type = 3
        self.nb_dense_block = nb_dense_block
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        # compute compression factor
        self.compression = 1.0 - reduction
    

    def create(self):
        # Initial convolution
        eps = 1.1e-5
        self.concat_axis = 3
        #if K.image_data_format() == 'channels_last':
        #    self.concat_axis = 3
        #    img_input = Input(shape=(self.img_rows, self.img_cols, self.color_type), name='data')
        #else:
        #    self.concat_axis = 1
        #    img_input = Input(shape=(self.color_type, self.img_rows, self.img_cols), name='data')
        x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(self.img_input)
        x = Conv2D(self.nb_filter, kernel_size=(7,7), strides=(2,2), name='conv1', use_bias=False)(x)
        x = BatchNormalization(epsilon=eps, axis=self.concat_axis, name='conv1_bn')(x)
        x = Scale(axis=self.concat_axis, name='conv1_scale')(x)
        x = Activation('relu', name='relu1')(x)
        x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

        # Add dense blocks
        for block_idx in range(self.nb_dense_block - 1):
            stage = block_idx+2
            x, self.nb_filter = self.dense_block(x, stage, self.nb_layers[block_idx], self.nb_filter, self.growth_rate, dropout_rate=self.dropout_rate, weight_decay=self.weight_decay)

            # Add transition_block
            x = self.transition_block(x, stage, self.nb_filter, compression=self.compression, dropout_rate=self.dropout_rate, weight_decay=self.weight_decay)
            self.nb_filter = int(self.nb_filter * self.compression)

        final_stage = stage + 1
        x, self.nb_filter = self.dense_block(x, final_stage, self.nb_layers[-1], self.nb_filter, self.growth_rate, dropout_rate=self.dropout_rate, weight_decay=self.weight_decay)

        x = BatchNormalization(epsilon=eps, axis=self.concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
        x = Scale(axis=self.concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
        x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

        x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
        x_fc = Dense(self.num_classes, name='fc6-f')(x_fc)
        # x_fc = Activation('softmax', name='prob-f')(x_fc)

        self.model = Model(self.img_input, x_fc, name='densenet')


        # Truncate and replace softmax layer for transfer learning
        # Cannot use model.layers.pop() since model is not of Sequential() type
        # The method below works since pre-trained weights are stored in layers but not in the model
        #x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
        #x_newfc = Dense(self.num_classes, name='fc6')(x_newfc)
        #x_newfc = Activation('softmax', name='prob')(x_newfc)
        
        #model = Model(self.img_input, x_newfc)
        self.output = x_fc

        # Learning rate is changed to 0.001
        # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        return self.model


    def load_weights(self):
        if self.densenet_size == 121:
            weights_path = 'imagenet_models/densenet121_weights_tf.h5'
        elif self.densenet_size == 161:
            weights_path = 'imagenet_models/densenet161_weights_tf.h5'
        elif self.densenet_size == 169:
            weights_path = 'imagenet_models/densenet169_weights_tf.h5'

        self.model.load_weights(weights_path, by_name=True)


    def conv_block(self, x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
        '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
            # Arguments
                x: input tensor 
                stage: index for dense block
                branch: layer index within each dense block
                nb_filter: number of filters
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''
        eps = 1.1e-5
        conv_name_base = 'conv' + str(stage) + '_' + str(branch)
        relu_name_base = 'relu' + str(stage) + '_' + str(branch)

        # 1x1 Convolution (Bottleneck layer)
        inter_channel = nb_filter * 4  
        x = BatchNormalization(epsilon=eps, axis=self.concat_axis, name=conv_name_base+'_x1_bn')(x)
        x = Scale(axis=self.concat_axis, name=conv_name_base+'_x1_scale')(x)
        x = Activation('relu', name=relu_name_base+'_x1')(x)
        # x = Conv2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False)(x)
        x = Conv2D(inter_channel, kernel_size=(1,1), name=conv_name_base+'_x1', use_bias=False)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        # 3x3 Convolution
        x = BatchNormalization(epsilon=eps, axis=self.concat_axis, name=conv_name_base+'_x2_bn')(x)
        x = Scale(axis=self.concat_axis, name=conv_name_base+'_x2_scale')(x)
        x = Activation('relu', name=relu_name_base+'_x2')(x)
        x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
        # x = Conv2D(nb_filter, 3, 3, name=conv_name_base+'_x2', bias=False)(x)
        x = Conv2D(nb_filter, kernel_size=(3,3), name=conv_name_base+'_x2', use_bias=False)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x


    def transition_block(self, x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
        ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_filter: number of filters
                compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''

        eps = 1.1e-5
        conv_name_base = 'conv' + str(stage) + '_blk'
        relu_name_base = 'relu' + str(stage) + '_blk'
        pool_name_base = 'pool' + str(stage) 

        x = BatchNormalization(epsilon=eps, axis=self.concat_axis, name=conv_name_base+'_bn')(x)
        x = Scale(axis=self.concat_axis, name=conv_name_base+'_scale')(x)
        x = Activation('relu', name=relu_name_base)(x)
        # x = Conv2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)
        x = Conv2D(int(nb_filter * compression), kernel_size=(1,1), name=conv_name_base, use_bias=False)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

        return x


    def dense_block(self, x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
        ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_layers: the number of layers of conv_block to append to the model.
                nb_filter: number of filters
                growth_rate: growth rate
                dropout_rate: dropout rate
                weight_decay: weight decay factor
                grow_nb_filters: flag to decide to allow number of filters to grow
        '''

        eps = 1.1e-5
        concat_feat = x

        for i in range(nb_layers):
            branch = i+1
            x = self.conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
            concat_feat = Concatenate(axis=self.concat_axis, name='concat_'+str(stage)+'_'+str(branch))([concat_feat, x])

            if grow_nb_filters:
                nb_filter += growth_rate

        return concat_feat, nb_filter

'''
if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 10 
    batch_size = 16 
    nb_epoch = 10

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

    # Load our model
    model = densenet121_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)
'''

if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 10 
    batch_size = 16 
    nb_epoch = 10

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

    # Load our model
    model_op = DenseNet(121, img_rows, img_cols, color_type=channel, num_classes=num_classes)
    model = model_op.create()

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)
