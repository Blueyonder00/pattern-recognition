import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

iris_df = datasets.load_iris()
x = iris_df.data  # x(150,4)
m = np.mean(x, axis=0)  # 按列求均值
M = x - (m.reshape(-1, 1).dot(np.ones((1, 150)))).T  # 每列减去均值
mcov = 1 / 150 * M.T.dot(M) # 求协方差矩阵
eig_value,eig_vector = np.linalg.eig(mcov) # 计算协方差矩阵的特征值和特征向量
eig_value_index = np.argsort(eig_value)[::-1]  # 将特征值从大到小排序，获得索引值
eig_value = eig_value[eig_value_index]  # 获得从大到小的特征值
eig_vector = eig_vector[:,eig_value_index]  # 获得和特征值排序相对应的特征向量

dx=(eig_vector[:,:2].T.dot(x.T)).T  # 降为两维
x1=dx[:50,:1]
y1=dx[:50,1:2]
x2=dx[50:100,:1]
y2=dx[50:100,1:2]
x3=dx[100:150,:1]
y3=dx[100:150,1:2]

# 画出降维后的图像
plt.scatter(x1,y1,c='r',label='Setosa')
plt.scatter(x2,y2,c='b',label='Versicolor')
plt.scatter(x3,y3,c='g',label='Vriginica')
plt.legend()
plt.show()


