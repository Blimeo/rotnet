import tensorflow as tf
import sys
from tf.keras.layers import Conv2D
from tf.keras.layers import BatchNormalization
from tf.keras.layers import Activation
from tf.keras.layers import MaxPooling2D

class ResNet():
    def __init__(self, training=True):
        self.training = training
        self.in_planes = 64

    def add_residual_block(self, input, block_number, in_channels, out_channels, downsample):
        block_number = str(block_number) #This was used for providing a unqiue name to each layer.
        skip = tf.identity(input)

        if downsample:
            skip = downsample(skip)

        x = Conv2D(filters=out_channels, name=block_number + "_1", kernel_size=(3, 3))(data)
        x = BatchNormalization()(x)
        x = tf.nn.relu()(x)

        x = Conv2D(filters=out_channels, name=block_number + "_2", kernel_size=(3, 3))(data)
        x = BatchNormalization()(x)

        x += skip

        return x

    def forward(self, data):
        #TODO: 64 7x7 convolutions followed by batchnorm, relu, 3x3 maxpool with stride 2
        data = Conv2D(filters=64, name='initial', kernel_size=(7, 7), input_shape=(32, 32, 3,))(data)
        data = BatchNormalization()(data)
        data = Activation('relu')(data)
        data = MaxPooling2D(pool_size=(3, 3), strides=2)(data)

        #TODO: Add residual blocks of the appropriate size. See the diagram linked in the README for more details on the architecture.
        # Use the add_residual_block helper function
        data = Conv2D(filters=64, name='conv3')

        #TODO: perform global average pooling on each feature map to get 4 output channels
        ...

        return logits

    def make_layer(self, out_channels, blocks, stride):
        if stride != 1 or self.inplanes != out_channels:
            downsample = Conv2D(self.in_planes, out_channels, stride)


        for _ in range(blocks):
            


    def add_convolution(self,
                        input,
                        name,
                        filter_size,
                        input_channels,
                        output_channels,
                        padding):
        return tf.nn.conv2d(input, filter=filter_size, padding=padding, strides=None, filters=output_channels, name=name, input_shape=input_channels)
        #first convolution - need a stride of 2
        #TODO: Implement a convolutional layer with the above specifications