import numpy as np


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