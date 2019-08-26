import tensorflow as tf
from tensorflow import keras

LAYER1_NODE = 512  # 第一层节点

LAYER2_NODE = 1024  # 第二层节点

LAYER3_NODE = 512  # 第三层节点

"""
input layer
layer1 512  tanh
layer2 1024  tanh
layer3 512  tanh
output layer sigmoid
"""


def dnn_interface(input_tensor, output_shape, regularizer_rate=None, drop=None):

    weight = []
    # 第一层密集层，添加L2正则
    with tf.variable_scope('dnn-layer1'):
        input_tensor_shape = input_tensor._shape_tuple()[1]
        # 第一层wight，shape需要设置
        layer1_weight = tf.get_variable('weight', [input_tensor_shape, LAYER1_NODE],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer_rate != None:
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer_rate)(layer1_weight))

        layer1_biase = tf.get_variable('biase', [LAYER1_NODE],
                                       initializer=tf.constant_initializer(0.0))

        layer1 = tf.nn.tanh(tf.matmul(input_tensor, layer1_weight) + layer1_biase)

        if drop != None:
            layer1 = tf.nn.dropout(layer1, drop)

        weight.append(layer1_weight)

    # 第二层密集层，添加L2正则
    with tf.variable_scope('dnn-layer2'):
        layer2_weight = tf.get_variable('weight', [LAYER1_NODE, LAYER2_NODE],
                                        initializer=tf.contrib.layers.xavier_initializer())

        if regularizer_rate != None:
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer_rate)(layer2_weight))

        layer2_biase = tf.get_variable('biase', [LAYER2_NODE],
                                       initializer=tf.constant_initializer(0.0))

        layer2 = tf.nn.tanh(tf.matmul(layer1, layer2_weight) + layer2_biase)

        if drop != None:
            layer2 = tf.nn.dropout(layer2, drop)

        weight.append(layer2_weight)

    # 第三层密集层，添加L2正则
    with tf.variable_scope('dnn-layer3'):
        layer3_weight = tf.get_variable('weight', [LAYER2_NODE, LAYER3_NODE],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer_rate != None:
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer_rate)(layer3_weight))

        layer3_biase = tf.get_variable('biase', [LAYER3_NODE],
                                       initializer=tf.constant_initializer(0.0))

        layer3 = tf.nn.tanh(tf.matmul(layer2, layer3_weight) + layer3_biase)

        if drop != None:
            layer3 = tf.nn.dropout(layer3, drop)

        weight.append(layer3_weight)
        
    # 输出层
    with tf.variable_scope('layer4_output'):
        layer4_weight = tf.get_variable('weight', [LAYER3_NODE, output_shape],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer_rate != None:
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer_rate)(layer4_weight))

        layer4_biase = tf.get_variable('biase', [output_shape],
                                       initializer=tf.constant_initializer(0.0))

        # sigmoid激活函数
        layer4 = tf.nn.sigmoid(tf.matmul(layer3, layer4_weight) + layer4_biase)

        weight.append(layer4_weight)
    
    return layer4, weight


def keras_dnn_interface(input_tensor, output_shape, regularizer_rate=None, drop=None):
    
    layer1 = keras.layers.Dense(128, activation='tanh', kernel_regularizer=regularizer_rate)(input_tensor)

    if drop is not None:
        layer1 = keras.layers.Dropout(drop)(layer1)
    
    layer2 = keras.layers.Dense(256, activation='tanh', kernel_regularizer=regularizer_rate)(layer1)
    
    if drop is not None:
        layer2 = keras.layers.Dropout(drop)(layer2)
    
    layer3 = keras.layers.Dense(128, activation='tanh', kernel_regularizer=regularizer_rate)(layer2)
    
    if drop is not None:
        layer3 = keras.layers.Dropout(drop)(layer3)
    
    y = keras.layers.Dense(output_shape, activation='sigmoid', kernel_regularizer=regularizer_rate)(layer3)
    
    return y

