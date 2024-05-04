# 载入数据集
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

x = iris.data
y = iris.target.reshape(-1, 1)
print(x.shape, y.shape)


#############################核心代码实现#############################


# 欧氏距离
def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# 分类器实现
class kNN(object):
    def __init__(self, n_neighbors=3, dist_func=distance):
        #  parameter: n_neighbors ,临近点个数; p ,距离度量; dist_func ,距离函数功能
        self.neighbors = n_neighbors
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


# 将数据分为训练集和测试集，用来测试模型分类正确率
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=1)

knn = kNN()
knn.fit(X_train, Y_train)

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

 #####################################  1. 不同k值对分类准确率的影响  ########################
# 定义一个knn实例
knn = kNN()

# 训练模型
knn.fit(X_train, Y_train)

result_list = []

# 考虑不同k值
for k in range(1, 10, 2):
    knn.neighbors = k
    # 传入测试数据，做预测
    Y_pred = knn.predict(X_test)
    # 统计预测正确的个数
    num_correct = np.sum(Y_pred == Y_test)
    # 计算准确率
    accuracy = float(num_correct) / X_test.shape[0]
    # 导入字典中
    result_list.append([k, accuracy])

pd.DataFrame(result_list, columns=['k', '预测准确率'])

############## 2. 采取另一种距离运算，是否会对分类结果造成影响(如曼哈顿距离)  曼哈顿距离 dist = |x_1-x_2|+|y_1-y_2| ###########


def distance2(x1, x2):
    return sum(abs(a - b) for a, b in zip(x1, x2))


knn = kNN(dist_func=distance2)
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

############################### 3. 不同的样本分布会对训练结果造成什么影响（附加题）###############################
##修改random_state，并分析为什么会导致训练准确率的改变？应如何准备样本，来确保训练结果收敛？


# 修改 random_state 可能会导致训练准确率的改变，因为random_state 参数通常用于确保代码每次运行时都能产生相同的结果。它是一个随机数种子，用于控制数据拆分的随机性。设置了 random_state，数据的拆分方式都会保持一致。它改变了训练集和测试集中的数据分布。不同的数据可能会影响模型的学习过程，从而影响准确率。
#
# 为了确保训练结果的收敛：
# 1.使用足够的数据量：更多的数据可以提供更全面的学习机会。
# 2.确保数据代表性：样本应该能够代表整个数据集的特征分布。
# 3.交叉验证：使用交叉验证可以减少模型对特定训练集的依赖。
# 4.调整 test_size：确保训练集和测试集的大小适当，以避免过拟合或欠拟合。
# 5.多次运行：通过改变 random_state 多次运行实验，然后平均结果，可以减少随机性对结果的影响
