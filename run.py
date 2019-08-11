import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from dnn import dnn_interface
from utils import *


INPUT_NODE = 480
OUTPUT_NODE = 480

dnn_drop = None # dnn的drop大小
dnn_regularizer_rate = None  # dnn的正则化大小
output_dnn_shape = None  # dnn输出节点数

LEARNING_RATE_BASE = 0.01 # 学习速率
BATCH_SIZE = 200  # 一批数据量
TRAIN_NUM = 20000  # 数据总量
TRAINING_STEPS = 500  # 训练多少次
SNR = []  # 要训练几个SNR

# 读取数据
X_train, X_test, Y_train, Y_test = read_data(0.25)


# 定义整个模型的x和y
x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x_input')

y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y_input')


# 神经网络
y, weight = dnn_interface(input_tensor=x,
                          output_shape=output_dnn_shape,
                          drop=dnn_drop,
                          regularizer_rate=dnn_regularizer_rate)

# 损失函数（需要修改）
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y,
                                                        labels=y_)  # 自动one-hot编码

cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 平均交叉熵

loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))  # 损失函数是交叉熵和正则化的和

# 优化器
# 定义当前迭代轮数的变量
global_step = tf.get_variable('global_step',  # 存储当前迭代的轮数
                              dtype=tf.int32,  # 整数
                              initializer=0,  # 初始化值
                              trainable=False)  # 不可训练

# 定义优化函数
train_step = tf.train.AdamOptimizer(LEARNING_RATE_BASE).minimize(loss, global_step)

with tf.Session() as sess:
    snr_loss = []
    for snr in SNR:
    
        E_x = 10 ** (0.1 * SNR)  # 信号能量
        
        # 添加噪声
        X = X * E_x + np.random.randn(TRAIN_NUM, INPUT_NODE, 2)  # sigma * r + mu
        ## 在前面归一化还是在后面归一化
        
        tf.global_variables_initializer().run()  # 初始化
        
        for i in range(TRAINING_STEPS):
            # 设置批次
            start = (i * BATCH_SIZE) % TRAIN_NUM
            end = min(start + BATCH_SIZE, TRAIN_NUM)

            _ = sess.run(train_step,
                         feed_dict={x: X_train[start:end], y_: Y_train[start:end]})
            
            train_loss = sess.run(loss,
                                  feed_dict={x: X_train[start:end], y_: Y_train[start:end]})
            
            test_loss = sess.run(loss,
                                 feed_dict={x: X_test[start:end], y_: Y_test[start:end]})

            if i % 100 == 0:
                print('snr：%d,训练了%d次,训练集损失%f,测试集损失%f' % (snr, i, train_loss, test_loss))
                
        #snr_loss.append(sess.run(loss, feed_dict={x: X_test[start:end], y_: Y_test[start:end]}))
    
    print('snr:', SNR)
    print('loss:', snr_loss)
