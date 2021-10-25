
import tensorflow as tf
from tensorflow.keras import initializers, Model
from tensorflow.keras.layers import Dense, BatchNormalization,Dropout, LeakyReLU

from app.contrib import ResNetTypeI

class Encoder(ResNetTypeI):
    def __init__(self):
        super(Encoder, self).__init__([2, 2, 2, 2]) #RESNET18 ResNetTypeI
#        super(Encoder, self).__init__([3, 4, 6, 3]) #RESNET50 ResNetTypeII

class Predictor(Model):
    def __init__(self):
        super(Predictor, self).__init__()
        initializer = initializers.HeNormal()
        self.Dense1 = Dense(16, activation='relu', 
                                    kernel_initializer=initializer)
        # self.bn1 = BatchNormalization()
        # self.leaky = LeakyReLU(alpha=0.2)
        self.dropout = Dropout(0.3)
        self.Dense2 = Dense(1, activation='sigmoid')

    def call(self, inputs, training = None, **kwargs):
        x = self.Dense1(inputs)
        x = self.bn1(x)
        x = self.leaky(x)
        x = self.dropout(x)
        x = self.Dense2(x)
        return x

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        initializer = tf.keras.initializers.HeNormal()

        self.Dense1 = Dense(16, activation='linear', 
                                    kernel_initializer=initializer)
        self.bn1 = BatchNormalization()
        self.leaky = LeakyReLU(alpha=0.2)
        self.dropout = Dropout(0.3)
        self.Dense2 = Dense(1, activation='sigmoid')

    def call(self, inputs, training = None, **kwargs):
        x = self.Dense1(inputs)
        x = self.bn1(x)
        x = self.leaky(x)
        x = self.dropout(x)
        x = self.Dense2(x)
        return x

    def condition(self,predicator_output, enc_output):
        p = tf.squeeze(predicator_output, axis=1)
        return tf.concat([p, enc_output], 1)
    
