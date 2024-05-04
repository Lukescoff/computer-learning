# 载入数据集
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

x = iris.data
y = iris.target.reshape(-1,1)

xf = pd.DataFrame(x)
yf = pd.DataFrame(y)

xf.to_csv('X.csv', index=False, header=False)
yf.to_csv('Y.csv', index=False, header=False)
print(x.shape, y.shape)


# 欧氏距离
def distance(x, x_text):
    num, num_label = x_text.shape
    ld = np.zeros(shape=(num, 1))
    sum_j = 0
    for i in range(num):
        for j in range(num_label):
            sum_j = sum_j + (x[j] - x_text[i][j]) ** 2
        ld[i] = np.sqrt(sum_j)
        sum_j = 0

    return ld


ran_x = np.array([5, 2, 3.4, 1])
dist = distance(ran_x, x)

dist_frame = pd.DataFrame(dist)
dist_frame.to_csv('dis.csv', index=False, header=False )
