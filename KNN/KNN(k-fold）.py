import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris

iris_df = load_iris()
x = iris_df.data
y = iris_df.target


def euclidean_distance(d1, d2):
    dist = np.sqrt(np.sum(np.power((d1 - d2), 2)))
    return dist


# 定义KNN
def KNN(train, test, k):
    true_num = 0
    for i in range(test.shape[0]):
        arr_dist = np.zeros((train.shape[0], 2))  # 初始化一个数组用于存放一个测试集样本和所有训练集样本的距离，
        # 第一列为距离，第二列为该训练样本的类别
        for j in range(train.shape[0]):
            dist = euclidean_distance(test[i, :-1], train[j, :-1])  # 计算每一个测试样本和训练样本的距离
            arr_dist[j, :] = dist, train[j, -1]  # 存放距离和类别
        arr_dist_df = pd.DataFrame(data=arr_dist, columns=['distance', 'target'])
        sort_df = arr_dist_df.sort_values(by='distance')['target'].head(k).value_counts()  # 对距离进行排序，选出前k个并按类别进行统计
        sort_value = sort_df.index[0]  # 选出前k个中出现最多的类别作为测试样本的类别
        if sort_value == test[i, -1]:  # 和测试样本所属原类别进行比较，统计预测正确的个数
            true_num += 1
    score = true_num / 30.0  # 计算score
    return score, true_num


k_list = list(range(1, 15))[::2]
score_var = []
kf = KFold(n_splits=5, random_state=0)  #
for k in k_list:
    score_list = []
    for train_index, test_index in kf.split(x, y):  #
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train = np.hstack((x_train, y_train.reshape(-1, 1)))
        test = np.hstack((x_test, y_test.reshape(-1, 1)))
        score, true = KNN(train, test, k)
        score_list.append(score)
    score_var.append(np.var(score_list))
    score_max = np.max(score_list)
    score_min = np.min(score_list)
    plt.scatter(k, score_max, color='r')
    plt.scatter(k, score_min, color='r')
plt.plot(k_list, score_var)
for a, b in zip(k_list, score_var):
    plt.text(a, b, "%.4f" % b, ha='center', va='bottom')
plt.xticks(k_list)
plt.show()
