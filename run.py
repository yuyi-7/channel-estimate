import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from dnn import *
from utils import *
from matplotlib import pyplot as plt
from datetime import datetime
INPUT_NODE = 480
OUTPUT_NODE = 480

dnn_drop = None # dnn的drop大小
dnn_regularizer_rate = None  # dnn的正则化大小

LEARNING_RATE_BASE = 0.001  # 学习速率
BATCH_SIZE = 100  # 一批数据量
TEST_SIZE = 0.25
TRAIN_NUM = 1e5  # 训练总量

TRAINING_STEPS = 10000  # 训练多少次
BER_CAL_NUM = 10000  # 评估用的数据量

STAIRCASE = True
SNR = [0, 10, 20, 30, 40]  # 要训练几个SNR

# # 读取数据
# X, Y = read_data()
#
# # 输出的归一化，只需要归一化一次
# Y = (Y - np.mean(Y)) / np.std(Y)

#计时开始
tic = datetime.now()

# 定义整个模型的x和y
x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x_input')

y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y_input')
print('定义tensor x和tensor y_')

# 神经网络

# y, weight, layer_output = dnn_interface(input_tensor=x,
#                                         output_shape=OUTPUT_NODE,
#                                         drop=dnn_drop,
#                                         regularizer_rate=dnn_regularizer_rate)

y = keras_dnn_interface(input_tensor=x,
                        output_shape=OUTPUT_NODE,
                        drop=dnn_drop)
print('定义神经网络输出 y')
# 损失函数
# loss = tf.math.divide(tf.reduce_sum(tf.abs(y - y_)), BATCH_SIZE * 480)  # BER

# 交叉熵
# cross_entropy = y_ * log(y) + (1-y_) * log(1-y)
y_sigmoid = tf.nn.sigmoid(y)
loss = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y_sigmoid, 1e-8, 1e2)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y_sigmoid, 1e-8, 1e2)))
print('定义损失函数 crossentropy')

# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,
#                                                                  logits=y))

# MSE
# loss = tf.reduce_mean(tf.square(y-y_))

# loss = ber
#loss = ber + tf.add_n(tf.get_collection('losses'))

# 优化器
# 定义当前迭代轮数的变量
global_step = tf.get_variable('global_step',  # 存储当前迭代的轮数
                              dtype=tf.int32,  # 整数
                              initializer=0,  # 初始化值
                              trainable=False)  # 不可训练

# 指数衰减学习速率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                           global_step,
                                           TRAIN_NUM/BATCH_SIZE,
                                           0.99,
                                           staircase=STAIRCASE)

# 定义优化函数
train_step = tf.train.AdamOptimizer(LEARNING_RATE_BASE).minimize(loss, global_step)
print('定义优化器 AdanOptimizer')

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配

with tf.Session(config=config) as sess:
    snr_ber = []
    train_loss_list = []
    test_loss_list = []
    for snr in SNR:
        train_loss_snr = []
        test_loss_snr = []

        print('开始读取数据...')
        X,Y = read_data(snr)
        print('读取数据完成，X:',X[:5],'\nY:',Y[:5])

        # 归一化
        X = (X - np.mean(X)) / np.std(X)
        print('完成 X 的归一化，归一化后mean:%.5f, std:%.5f'%(X.mean(),X.std()))
        # Y = (Y - np.mean(Y)) / np.std(Y)
        # print(X)
        # print('mean: %f, std: %f'%(X.mean(),X.std()))
        # print(Y)

        # 分离训练集与测试集
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE)
        print('完成训练集测试集分离，测试集占比%f'%TEST_SIZE)

        tf.global_variables_initializer().run()  # 初始化
        
        for i in range(TRAINING_STEPS):
            # 设置批次

            start = int((i * BATCH_SIZE) % TRAIN_NUM)
            end = int(min(start + BATCH_SIZE, TRAIN_NUM))

            # print('start:%d, end:%d'%(start,end))

            sess.run(train_step,
                     feed_dict={x: X_train[start:end], y_: Y_train[start:end]})

            train_test_ = int(TRAIN_NUM * (1-TEST_SIZE))
            if start < train_test_:
                train_loss = sess.run(loss,
                                      feed_dict={x: X_train[start:end], y_: Y_train[start:end]})
            
            if start >= train_test_:
                test_loss = sess.run(loss,
                                     feed_dict={x: X_test[start-train_test_:end-train_test_],
                                                y_: Y_test[start-train_test_:end-train_test_]})

            if i % 100 == 0:

                if start >= train_test_:
                    print('snr：%d,训练了%d次,训练集损失%.12f,测试集损失%.12f' % (snr, i, train_loss, test_loss))
                    train_loss_snr.append(train_loss)
                    test_loss_snr.append(test_loss)
                    # print('x:')
                    # print(sess.run(x,feed_dict={x: X_test[start-25000:start-25000+1]}))
                    # print('y:')
                    # print(sess.run(y,feed_dict={x: X_test[start-25000:start-25000+1]}))
                    # print(sess.run(y_,feed_dict={y_:Y_test[start-25000:start-25000+1]}))
                else:
                    print('snr：%d,训练了%d次,训练集损失%.12f' % (snr, i, train_loss))
                    train_loss_snr.append(train_loss)
                # print('layer_output:')
                # print(sess.run(layer_output,feed_dict={x: X_train[start:start+1], y:Y_train[start:start+1]}))
                # print('y_:')
                # print(sess.run(y_,feed_dict={y_:Y_train[start:start+1]}))
                # print('layer4:')
                # print(sess.run(y,feed_dict={x:X_train[start:start+1]}))

        train_loss_list.append(train_loss_snr)
        test_loss_list.append(test_loss_snr)
        # print(test_loss_list)

        # 开始评估
        # 映射到0，1

        ber_y = tf.where(tf.greater(y, tf.zeros_like(y)), tf.ones_like(y), tf.zeros_like(y))
        # ber_y_ = tf.where(tf.greater_equal(y_, tf.zeros_like(y_)), tf.ones_like(y), tf.zeros_like(y))

        # 评估数据的index
        index = np.random.randint(0, int(TRAIN_NUM), BER_CAL_NUM)

        
        # 用1000帧数据评估ber
        snr_present = sess.run(
                tf.math.divide(tf.reduce_sum(tf.abs(ber_y - y_)), BER_CAL_NUM * 480),
                feed_dict={x: np.array(X)[index],
                           y_: np.array(Y)[index]})
        print('y:',end='\t')
        print(sess.run(y, feed_dict={x: np.array(X)[index]}))
        print('y_softmax:', end='\t')
        print(sess.run(y_sigmoid, feed_dict={x: np.array(X)[index]}))
        print('ber_y:', end='\t')
        print(sess.run(ber_y, feed_dict={x: np.array(X)[index]}))
        print('y_:', end='\t')
        print(sess.run(y_, feed_dict={y_: np.array(Y)[index]}))

        print(snr_present)
        snr_ber.append(snr_present)

    # snr_ber = np.log10(snr_ber)
    print('snr:', SNR)
    print('loss:', snr_ber)
    toc = datetime.now()#计时结束
    print('Elapsed time:%f seconds'%(toc - tic).total_seconds())#计算本次运行时间


    plt.figure()
    plt.plot(SNR, snr_ber)
    plt.grid()
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.title('Deep Learning on channel estimate')
    plt.show()

    pd.DataFrame(data=train_loss_list, index=SNR).T.plot()
    plt.title('Train loss')
    plt.show()

    pd.DataFrame(data=test_loss_list, index=SNR).T.plot()
    plt.title('Test loss')
    plt.show()
