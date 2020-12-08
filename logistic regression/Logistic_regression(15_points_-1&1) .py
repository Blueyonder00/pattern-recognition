'''SGD'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

data = pd.read_csv('15points.csv', header=None, names=['x1', 'x2', 'y'])  # 以dataframe的形式读取数据
x_ = data.iloc[:, 0:2].values  # 取出前两列作为x并转换为矩阵 x(15,2)
x = np.array(np.hstack((x_, np.ones((x_.shape[0], 1)))))  # 为x加上一列1，x(15,3)
y = np.array(data.iloc[:, -1].values.reshape(-1, 1)) # 取出最后一列最为y并转换为矩阵 y(15,1)
loss = []
# 定义sigmoid函数形式
def sigmoid(z):
    S = 1 / (1 + np.exp(-z))
    return S
# 定义损失函数
def cross_entropy(x0, y0, B0):
    z1 = -np.log(sigmoid(np.multiply(y0, x0.dot(B0))))
    return np.sum(z1)/15
# 对数据重新洗牌
def suffledata(data):
    data = shuffle(data)
    x_1 = data.iloc[:, 0:2].values
    x = np.array(np.hstack((x_1, np.ones((x_1.shape[0], 1)))))
    y = np.array(data.iloc[:, -1].values.reshape(-1, 1))
    return x,y

# 随机梯度下降法
def w_calu(data, r=0.01, iter=20000):
    B = np.array(np.random.randn(3,1))  # 初始化B0，随机生成满足(0,1)高斯分布的权值
    for i in range(iter):
        x,y = suffledata(data)  # 对数据重新洗牌
        j = np.random.randint(0,15)   # SGD
        H = sigmoid(-y[j,0] * x[j,:].dot(B))
        deltaB = -(H * y[j,0] * x[j,:]).reshape(-1,1) # 计算梯度方向
        B -= r * deltaB # 更新梯度
        loss.append(cross_entropy(x,y,B))
    return B,loss


B,loss = w_calu(data, 0.01,iter=30000)   #调用函数计算权值
x1=np.array(x[:,0]).reshape(-1,1)
x2=np.array(x[:,1]).reshape(-1,1)
print('B=',B)

plt.subplot(221)
plt.scatter(x1[y == -1], x2[y == -1], c='b', label='-1')
plt.scatter(x1[y == 1], x2[y == 1], c='r', label='1')
plt.grid()
plt.legend()

plt.subplot(222)
plt.scatter(x1[y == -1], x2[y == -1], c='b', label='-1')
plt.scatter(x1[y == 1], x2[y == 1], c='r', label='1')
x_axis1 = np.arange(0, 12)
y_axis1 = (-B[0, 0] * x_axis1 - B[2, 0]) / B[1, 0]
plt.plot(x_axis1, y_axis1, c='black')
plt.grid()
plt.legend()

plt.subplot(212)
x_axis2 = np.arange(1,30001)
plt.plot(x_axis2,loss,c='b')
plt.show()
