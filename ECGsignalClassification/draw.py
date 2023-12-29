import pydot

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import plot_model

newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(300,)),
        # reshape the tensor with shape (batch_size, 300) to (batch_size, 300, 1)
        tf.keras.layers.Reshape(target_shape=(300, 1)),
        # the first convolution layer, 4 21x1 convolution kernels, output shape (batch_size, 300, 4)
        tf.keras.layers.Conv1D(filters=4, kernel_size=10, strides=1, padding='SAME', activation='relu'),
        # the first pooling layer, max pooling, pooling size=3 , stride=2, output shape (batch_size, 150, 4)
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # the second convolution layer, 16 23x1 convolution kernels, output shape (batch_size, 150, 16)
        tf.keras.layers.Conv1D(filters=16, kernel_size=10, strides=1, padding='SAME', activation='relu'),
        # the second pooling layer, max pooling, pooling size=3, stride=2, output shape (batch_size, 75, 16)
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # the third convolution layer, 32 25x1 convolution kernels, output shape (batch_size, 75, 32)
        tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='SAME', activation='relu'),
        # the third pooling layer, average pooling, pooling size=3, stride=2, output shape (batch_size, 38, 32)
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME'),
        # the fourth convolution layer, 64 27x1 convolution kernels, output shape (batch_size, 38, 64)
        tf.keras.layers.Conv1D(filters=64, kernel_size=10, strides=1, padding='SAME', activation='relu'),
        # flatten layer, for the next fully connected layer, output shape (batch_size, 38*64)
        tf.keras.layers.Flatten(),
        # fully connected layer, 128 nodes, output shape (batch_size, 128)
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout layer, dropout rate = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        # fully connected layer, 5 nodes (number of classes), output shape (batch_size, 5)
        tf.keras.layers.Dense(5, activation='softmax')
    ])

# 创建一个Sequential模型
model = Sequential()

# 添加卷积层和池化层
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# 展平层
model.add(Flatten())

# 添加全连接层
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))  # 假设有5个类别

# 打印模型概述
model.summary()

# 保存模型结构图到TensorBoard日志目录
plot_model(newModel, to_file='cnn_model.png', show_shapes=True)
