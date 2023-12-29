import tensorflow as tf

from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.models import Model


def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    # First convolutional layer
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Second convolutional layer
    x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    # Shortcut connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride)(shortcut)
    x = tf.keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet18(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Residual blocks
    num_blocks_list = [2, 2, 2, 2]
    filters_list = [64, 128, 256, 512]
    for stage in range(4):
        for block in range(num_blocks_list[stage]):
            stride = 1 if stage == 0 and block == 0 else 2
            x = residual_block(x, filters_list[stage], stride=stride)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Define input shape and number of classes
input_shape = (224, 224, 3)
num_classes = 1000

# Create the ResNet-18 model
resnet18 = ResNet18(input_shape, num_classes)

# Print model summary
resnet18.summary()
