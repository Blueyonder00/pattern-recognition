import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 导入iris数据集，取第一类花的”sepal length"为输入变量，“sepal width”为输出变量
iris_df = datasets.load_iris()
x = iris_df['data'][:, 2:3]
y = iris_df['data'][:, 3:4]
Data = np.hstack((x, np.ones((x.shape[0], 1))))  # 输入矩阵右侧增加一列1，用来对应系数beta0


B0 = [[1], [0]]  # 初始化B0
iter = 0  # 迭代次数
n = x.shape[0]
r = 0.01  # 学习率

while iter < 10000:
    deltaB = 2 / n * (-Data.T.dot(y) + Data.T.dot(Data).dot(B0))  # 计算梯度方向
    B0 = B0 - r * deltaB  # 更新梯度
    iter += 1  # 迭代次数加1
    loss=1/n*np.linalg.norm(y.reshape(x.shape[0],1)-Data.dot(B0))**2
    if loss<0.01:
        break
# 绘制散点图
plt.subplot(211)
plt.scatter(x, y, c='b')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
# 绘制拟合曲线
plt.subplot(212)
plt.scatter(x, y, c='b')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
y_ = np.dot(Data, B0)
plt.plot(x, y_, c='black')
plt.show()

print(iter,loss)