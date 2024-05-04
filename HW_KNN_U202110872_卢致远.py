#!/usr/bin/env python
# coding: utf-8

# # 第一次作业
# 根据课堂所学KNN原理，补全KNN核心运算代码
# 
# 并在本文件中实现功能，回答以下问题
# 
# 1. 不同k值对分类准确率的影响
# 2. 采取另一种距离运算，是否会对分类结果造成影响(如曼哈顿距离)
# 3. 不同的样本分布会对训练结果造成什么影响（附加题）

# ### Markdown基础语法
# 将该代码框调整为Markdown格式即可进行文本编辑
# ![image.png](attachment:image.png)
# 
# 基础语法参考链接：https://www.jianshu.com/p/191d1e21f7ed/
# 

# In[54]:


# 载入数据集
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


# In[55]:


iris = datasets.load_iris()

x = iris.data
y = iris.target.reshape(-1,1)
print(x.shape,y.shape)


# In[56]:


#############################核心代码实现#############################

# 欧氏距离
def distance(x_unknown, x_train):
    return np.sqrt(np.sum((x_unknown - x_train) ** 2))

# 分类器实现
class kNN():
    def  __init__(self, n_neighbors=3, p=2, dist_func=distance):
         #  parameter: n_neighbors ,临近点个数; p ,距离度量; dist_func ,距离函数功能
        self.neighbors = n_neighbors
        self.p = p
        self.dist_func = dist_func
        
        
    def fit(self, x, y):
         # 将训练集传进来即可
        self.x_train = x
        self.y_train = y
        
        
    def predict(self, x_test):
        predict_list = []
        for i in range(np.shape(x_test)[0]):
            x = x_test[i]
            # 计算距离
            distances = [self.dist_func(x, x_train) for x_train in self.x_train]
            # 获取最近的k个样本的索引
            k_indices = np.argsort(distances)[:self.neighbors]
            # 获取这些样本的类别标签
            k_nearest_labels = [self.y_train[j] for j in k_indices]
            # 多数投票
            most_common = self.vote_count(k_nearest_labels)
            predict_list.append(most_common)
        # 转置一下，保证格式与Y_test 一样
        predict_list = np.array(predict_list).reshape(-1, 1)
        return predict_list

    
    
    def vote_count(self, labels):
        # 计票
        label_count = {}
        for label_array in labels:
            label = label_array[0]      # 这里是因为输入的标签是已经是矩阵[[1],[0],[1]...]的形式（而不是字符串），矩阵label没法作为键
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1
        # 排序并返回出现次数最多的标签
        sorted_label_count = sorted(label_count.items(), key=lambda item: item[1], reverse=True)
        return sorted_label_count[0][0]


# In[57]:


# 将数据分为训练集和测试集，用来测试模型分类正确率
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state = 1)


# In[58]:


knn = kNN()
knn.fit(X_train,Y_train)

# 传入测试数据，做预测
Y_pred = knn.predict(X_test)
print('Prediction: ', Y_pred)

# 实际结果
print('Test value: ', Y_test)

# 统计预测正确的个数
num_correct = np.sum(Y_pred == Y_test)

# 计算准确率
accuracy = float(num_correct) / X_test.shape[0]
print('Got %d / %d correct => accuracy: %f' % (num_correct, X_test.shape[0], accuracy))


# ### 1. 不同k值对分类准确率的影响

# In[61]:


# 定义一个knn实例
knn = kNN()

# 训练模型,用已经划分好的训练集
knn.fit(X_train,Y_train)

result_list = []

#考虑不同k值
for k in range(1,10,2):
    knn.neighbors = k
    # 传入测试数据，做预测
    Y_pred = knn.predict(X_test)
    # 统计预测正确的个数
    num_correct = np.sum(Y_pred == Y_test)
    # 计算准确率
    accuracy = float(num_correct) / X_test.shape[0]
    # 导入字典中
    result_list.append([k, accuracy])

pd.DataFrame(result_list,columns = ['k','预测准确率'])


#   一般来说，较小的k值意味着，对近邻的依赖更强。这可能使得模型对训练数据有很好的预测，但可能不适用于未见过的数据，或者有一些异常值。相反，较大的k值提供了更平滑的决策边界，对噪声和异常值不那么敏感。
# 
#   上面的k不断变化但准确率一直不变的原因可能是：样本数量不足，有一个异常样本在特征空间中远离了其他同类样本。
# 

# ### 2. 采取另一种距离运算，是否会对分类结果造成影响(如曼哈顿距离)
# $$
# 曼哈顿距离\qquad dist = |x_1-x_2|+|y_1-y_2|
# $$

# In[60]:


def distance2(x1, x2):
    return sum(abs(a - b) for a, b in zip(x1, x2))

knn = kNN(dist_func = distance2)
knn.fit(X_train, Y_train)

# 传入测试数据，做预测
Y_pred2 = knn.predict(X_test)
print('Prediction: ', Y_pred2)

# 实际结果
print('Test value: ', Y_test)

# 统计预测正确的个数
num_correct = np.sum(Y_pred2 == Y_test)

# 计算准确率
accuracy = float(num_correct) / X_test.shape[0]
print('Got %d / %d correct => accuracy: %f' % (num_correct, X_test.shape[0], accuracy))


# 就本题给出的数据集来看，结果没有变化。
# 
# 但是，本题只是刚好结果一致。采用不同的距离度量（曼哈顿距离），确实会对KNN算法的分类结果产生影响。KNN算法中的距离度量用于计算数据点之间的相似性，不同的距离度量可能会导致不同的分类决策，进而影响结果。**如果数据集中的特征存在不同的尺度或单位**，欧氏距离可能会被高尺度的特征所主导，这时候曼哈顿距离可能是更好的选择。相反，**如果数据点是均匀分布的**，欧氏距离可能会提供更好的准确率。

# ### 3. 不同的样本分布会对训练结果造成什么影响（附加题）
# 修改random_state，并分析为什么会导致训练准确率的改变？应如何准备样本，来确保训练结果收敛？

# 修改 random_state 可能会导致训练准确率的改变，因为random_state 参数通常用于确保代码每次运行时都能产生相同的结果，它是一个随机数种子，用于控制数据拆分的随机性，它改变了那么训练集和测试集中的数据分布就改变了。不同的数据可能会影响模型的学习过程，从而影响准确率。
# 
# 为了确保训练结果的收敛：
# 
# 1.使用足够的数据量：**更多的数据**可以提供更全面的学习机会，调整 test_size。
# 
# 2.确保数据代表性：样本应该能够代表整个数据集的特征分布。
# 
# 3.交叉验证：使用交叉验证可以减少模型对特定训练集的依赖。
# 
# 4.改变 random_state： 多次运行实验，然后**平均结果**，可以减少随机性对结果的影响

# In[ ]:




