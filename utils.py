import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# 把虚部放在一起，实部放在一起,顺便归一化
def reshape_dim(a):
    temp = []
    a_mean = np.mean(a)
    a_std = np.std(a)

    for i in range(np.array(a).shape[0]):
        temp1 = []
        temp2 = []

        for j in a[i]:
            j_0 = (j[0] - a_mean) / a_std
            j_1 = (j[0] - a_mean) / a_std
            temp1.append(j_0)
            temp2.append(j_1)
        temp1.extend(temp2)
        temp.append(temp1)
    return np.array(temp).astype(np.float32)


def read_data(test_size):
    # 读取数据
    data_output_imag = pd.read_csv('0dB_source_data_imag.csv', header=None)
    data_output_real = pd.read_csv('0dB_source_data_real.csv', header=None)
    data_input_imag = pd.read_csv('0dB_zf_data_imag.csv', header=None)
    data_input_real = pd.read_csv('0dB_zf_data_real.csv', header=None)
    
    # 分开输入输出,实部虚部放一起
    X = pd.concat([data_input_real, data_input_imag], axis=1)
    Y = pd.concat([data_output_real, data_output_imag], axis=1)
    
    # 训练测试数据分离
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    
    return X_train, X_test, Y_train, Y_test