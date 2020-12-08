import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from scipy import stats

iris_df = load_iris()
x = iris_df.data
y = iris_df.target


# 计算先验概率
def prior(y_train):
    P0 = sum([i == 0 for i in y_train]) / float(len(y_train))
    P1 = sum([i == 1 for i in y_train]) / float(len(y_train))
    P2 = sum([i == 2 for i in y_train]) / float(len(y_train))
    Py = np.array([P0, P1, P2])
    return Py


# 将每一类中每一个特征都看成高斯分布，求出均值和方差
def Gaussian(x_train, y_train):
    mu = np.zeros((3, 4))
    sd = np.zeros((3, 4))
    for i in range(3):
        for j in range(x_train.shape[1]):
            sample = x_train[y_train == i, j]
            mu[i, j] = np.mean(sample)
            sd[i, j] = np.std(sample)
    return mu, sd


# 预测样本
def pred(x_test, y_train, mu, sd):
    Pxy = np.ones((30, 3))
    for i in range(30):
        for j in range(3):
            for k in range(4):  # 求P(xi|y)
                Pxy[i, j] *= stats.norm(mu[j, k], sd[j, k]).pdf(x_test[i, k])
            Pxy[i, j] *= prior(y_train)[j]  # P(x1|y)P(x2|y)...P(xi|y)P(y) 其实先验相同，可以直接比较似然的大小
    maxPxy = np.argmax(Pxy, axis=1).reshape(-1, 1)
    return maxPxy


# 将数据集打乱随机分层划分测试集和训练集，训练集中各类比例一致
ss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=None)
for train_index, test_index in ss.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    mu, sd = Gaussian(x_train, y_train)
    maxPxy = pred(x_test, y_train, mu, sd)
    true_num = 0
    for i in range(30):
        if maxPxy[i] == y_test[i]:
            true_num += 1
    score = true_num / 30.0
    print('准确度为：%.6f' % score)
