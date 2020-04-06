import tensorflow as tf
import sys

class ResNet():
    def __init__(self, training=True):
        self.training = training

    def add_residual_block(self, input, block_number, in_channels, out_channels):
        block_number = str(block_number) #This was used for providing a unqiue name to each layer.
        skip = tf.identity(input)
        # After each residual block, the number of channels doubles while the feature map is halved in dimension.
        # We downsample the feature map through stride.
        down = 1
        if in_channels != out_channels: 
            skip = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=2, padding='same')(skip)
            skip = tf.keras.layers.BatchNormalization()(skip)
            down = 2

        x = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, padding="same", strides=down)(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, skip])
        x = tf.nn.relu(x)

        return x

    def forward(self, data):
        #TODO: 64 7x7 convolutions followed by batchnorm, relu, 3x3 maxpool with stride 2
        data = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, input_shape=(32, 32, 3,), padding="same")(data)
        data = tf.keras.layers.BatchNormalization()(data)
        data = tf.keras.layers.Activation('relu')(data)
        data = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(data)

        #TODO: Add residual blocks of the appropriate size. See the diagram linked in the README for more details on the architecture.
        # Use the add_residual_block helper function
        # 
        data = self.add_residual_block(input=data, block_number='2_1', in_channels=64, out_channels=64)
        data = self.add_residual_block(input=data, block_number='2_2', in_channels=64, out_channels=64)
        data = self.add_residual_block(data, '3_1', 64, 128)
        data = self.add_residual_block(data, '3_2', 128, 128)
        data = self.add_residual_block(data, '4_1', 128, 256)
        data = self.add_residual_block(data, '4_2', 256, 256)
        data = self.add_residual_block(data, '5_1', 256, 512)
        data = self.add_residual_block(data, '5_2', 512, 512)
       
        #TODO: perform global average pooling on each feature map to get 4 output channels

        data = tf.keras.layers.GlobalAveragePooling2D()(data)
        data = tf.keras.layers.Dense(4)(data)
        return data