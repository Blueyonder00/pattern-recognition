import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('15points.csv', header=None, names=['x1', 'x2', 'y'])  # 以dataframe的形式读取数据
x_ = data.iloc[:, 0:2].values  # 取出前两列作为x并转换为矩阵 x(15,2)
x = np.array(np.hstack((x_, np.ones((x_.shape[0], 1)))))
y_= data.iloc[:, -1].values.reshape(-1, 1)  # 取出最后一列最为y并转换为矩阵 y(15,1)
y = np.array([1 if i == 1 else 0 for i in y_]).reshape(-1, 1)

def sigmoid(z):
    S = 1 / (1 + np.exp(-z))
    return S

# 计算cost function
def cross_entropy(x0, y0, B0):
    a1 = -np.multiply(y0, np.log(sigmoid(x0.dot(B0))))
    a2 = np.multiply((1 - y0), np.log(1- sigmoid(x0.dot(B0))))
    return np.sum(a1 - a2) / 15

def w_calu(x, y, r=0.01, iter=10000):
    cost = []
    B = np.array(np.random.randn(3,1))  # 初始化B0，随机生成满足(0,1)高斯分布的权值
    for i in range(iter):
        S = sigmoid(x.dot(B))
        deltaB = x.T.dot(S - y)
        B -= r * deltaB
        cost.append(cross_entropy(x,y,B))
    return B,cost


B,cost = w_calu(x, y, 0.01, 50000)
print('B=', B)
x1=np.array(x[:,0]).reshape(-1,1)
x2=np.array(x[:,1]).reshape(-1,1)

plt.subplot(221)
plt.scatter(x1[y == 0], x2[y == 0], c='b', label='0')
plt.scatter(x1[y == 1], x2[y == 1], c='r', label='1')
plt.grid()
plt.legend()

plt.subplot(222)
plt.scatter(x1[y == 0], x2[y == 0], c='b', label='0')
plt.scatter(x1[y == 1], x2[y == 1], c='r', label='1')
x_axis = np.arange(0, 13)
y_axis = (-B[0, 0] * x_axis - B[2, 0]) / B[1, 0]
plt.plot(x_axis, y_axis, c='black')
plt.grid()
plt.legend()

plt.subplot(212)
x2 = np.arange(1,50001)
plt.plot(x2,cost,c='b')
plt.show()

