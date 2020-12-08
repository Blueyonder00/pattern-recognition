import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
# 导入iris数据集，取第一类花的”sepal length"为输入变量，“sepal width”为输出变量
iris_df = datasets.load_iris()
x = iris_df['data'][0:50, 0:1]
y = iris_df['data'][0:50, 1:2]

Data = np.hstack((x, np.ones((x.shape[0], 1))))  # 输入矩阵右侧增加一列1，用来对应系数beta0

B = np.dot(np.dot(np.linalg.inv(np.dot(Data.T, Data)), Data.T), y) #运用公式求解参数矩阵B
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
y_ = np.dot(Data, B)
print(y_)
plt.plot(x, y_, c='black')
plt.show()
