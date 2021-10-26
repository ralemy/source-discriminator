# Source: https://github.com/calmisential/TensorFlow2.0_ResNet 
# refactored to remove the predictor to its own model and keep only code enough for a RESNET50 Model
# reduced to RESNET18 after running out of resources.

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalAveragePooling2D, Layer, Flatten

class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1, kernel_size = (3,3)):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=kernel_size,
                                            strides=stride,
#                                            kernel_regularizer=tf.keras.regularizers.l2(l2),
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=kernel_size,
                                            strides=1,
 #                                           kernel_regularizer=tf.keras.regularizers.l2(l2),
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       kernel_regularizer=tf.keras.regularizers.l2(l2),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output



class BottleNeck(Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = Conv2D(filters=filter_num, kernel_size=(1, 1), strides=1, padding='same')
        self.bn1 = BatchNormalization()
        
        self.conv2 = Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding='same')
        self.bn2 = BatchNormalization()
        
        self.conv3 = Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=1, padding='same')
        self.bn3 = BatchNormalization()

        self.downsample = tf.keras.Sequential()
        self.downsample.add(Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=stride))
        self.downsample.add(BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output

def make_basic_block_layer(filter_num, blocks, stride=1, kernel_size=(3,3)):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride, kernel_size=kernel_size))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1, kernel_size=kernel_size))

    return res_block

def make_bottleneck_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1))

    return res_block

class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.Flatten = Flatten()

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.Flatten(x)
        return output

class ResNetType0(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetType0, self).__init__()


        self.layer1 = make_basic_block_layer(filter_num=8,
                                             blocks=layer_params[0], kernel_size= (12,12))
        self.layer2 = make_basic_block_layer(filter_num=16,
                                             blocks=layer_params[1],
                                             stride=2, kernel_size=(6,6))
        self.layer3 = make_basic_block_layer(filter_num=32,
                                             blocks=layer_params[2],
                                             stride=2, kernel_size=(6,6))
        self.layer4 = make_basic_block_layer(filter_num=32,
                                             blocks=layer_params[3],
                                             stride=2, kernel_size=(3,3))

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.Flatten = Flatten()

    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.Flatten(x)
        return output

class ResNetTypeX(tf.keras.Model): #Custom added type
    def __init__(self, layer_params):
        super(ResNetTypeX, self).__init__()

        # input 64 * 8192 * 1
        self.layer1 = make_basic_block_layer(filter_num=8,
                                             blocks=layer_params[0], 
                                             stride=2, kernel_size= (5,5))
        self.layer2 = make_basic_block_layer(filter_num=16,
                                             blocks=layer_params[1],
                                             stride=2, kernel_size=(3,3))
        self.layer3 = make_basic_block_layer(filter_num=32,
                                             blocks=layer_params[2],
                                             stride=2, kernel_size=(3,3))
        self.layer4 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[3],
                                             stride=2, kernel_size=(3,3))
        self.layer4 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[4],
                                             stride=2, kernel_size=(3,3))
        self.layer4 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[5],
                                             stride=2, kernel_size=(3,3))
        # output 1 * 128 * 256
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.Flatten = Flatten()

    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.Flatten(x)
        return output


class ResNetTypeII(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeII, self).__init__()
        self.conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same")
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPool2D(pool_size=(3, 3), strides=2, padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64, blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128, blocks=layer_params[1], stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256, blocks=layer_params[2], stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,blocks=layer_params[3],stride=2)

        self.avgpool = GlobalAveragePooling2D()
        self.Flatten = Flatten()


    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.Flatten(x)
        return output

