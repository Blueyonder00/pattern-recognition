"""
Batch Gradient Descent
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris_df = datasets.load_iris()  # 读取iris数据集
x = iris_df.data[:100, :2]  # data对应样本的4个特征（150行4列），取出数据集前两类品种（100行）的前两列特征作为输入
y = iris_df.target[:100]  # target对应样本的类别属性（150行1列）
y = np.array([1 if i == 1 else -1 for i in y]).reshape(-1, 1)  # 将原始数据集前两类的输出转换为+1，-1二值

Data = np.hstack((x, np.ones((x.shape[0], 1))))  # 输入矩阵右侧增加一列1，用来对应系数beta0

iter = 0
B0 = np.array([[0], [0], [0]])  # 权值初始化
r = 0.001  # 学习率
mistaken = -np.multiply(np.dot(Data, B0), y)
num = np.sum(mistaken>=0)  # 判断错误分类的点的个数
dc = mistaken >= 0  # 为错误分类的点打上“true"的标签
error = []

while num > 0 and iter < 10000:

    deltaB = np.dot(Data[dc.reshape(Data.shape[0], ), :].T, y[dc.reshape(Data.shape[0], )].reshape(-1, 1))  # 批量梯度下降法计算梯度
    B0 = B0 + r * deltaB.reshape(-1, 1)  # 更新B0
    iter += 1
    mistaken = []
    dc=[]
    mistaken = -np.multiply(np.dot(Data, B0), y)
    num = np.sum(mistaken > 0)
    dc = mistaken > 0
    error_single = -np.dot(np.dot(Data[dc.reshape(Data.shape[0],), :], B0).T,
                           y[dc.reshape(Data.shape[0],)].reshape(-1, 1)) / np.linalg.norm(B0)
    error.append(error_single.reshape(1, ).tolist())
# 绘制散点图
plt.subplot(221)
plt.scatter(x[:50, :1], x[:50, 1:2], c='r')  #
plt.scatter(x[50:100, :1], x[50:100, 1:2], c='b')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
# 绘制分类后的图形
plt.subplot(222)
plt.scatter(x[:50, :1], x[:50, 1:2], c='r')
plt.scatter(x[50:100, :1], x[50:100, 1:2], c='b')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
x_ = np.arange(4, 8)
y_ = (-B0[0, 0] * x_ - B0[2, 0]) / B0[1, 0]
plt.plot(x_, y_, c='g')

plt.subplot(212)
x_points = np.arange(1,101).reshape(-1,1)
plt.plot(x_points,error[0:100],c='b')
plt.show()

print(iter)