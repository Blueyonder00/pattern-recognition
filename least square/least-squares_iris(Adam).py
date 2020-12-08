import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 导入iris数据集，取第一类花的”petal length"为输入变量，“petal width”为输出变量
iris_df = datasets.load_iris()
x = iris_df['data'][:, 2:3]
y = iris_df['data'][:, 3:4]
Data = np.hstack((x, np.ones((x.shape[0], 1))))  # 输入矩阵右侧增加一列1，用来对应系数beta0

B0 = [[1], [0]]  # 初始化B0
iter = 50000  # 迭代次数
n = Data.shape[0]
r = 0.01  # 学习率
beta1 = 0.9  # 算法作者建议的默认值
beta2 = 0.999  # 算法作者建议的默认值
eps = 1e-8  # 算法作者建议的默认值
mt = np.zeros((Data.shape[1],1))
vt = np.zeros((Data.shape[1],1))
loss = []

for i in range(iter):
    j = np.random.randint(0,149)
    loss.append(1 / n * np.linalg.norm(y.reshape(x.shape[0], 1) - Data.dot(B0))**2)
    if loss[i] < 0.01:
        break
    deltaB = ((Data[j].dot(B0) - y[j]) * Data[j].T).reshape(-1,1) # 计算梯度方向
    mt = beta1 * mt + (1 - beta1) * deltaB
    vt = beta2 * vt + (1 - beta2) * (deltaB ** 2)
    mtt = mt / (1 - (beta1 ** (i + 1)))
    vtt = vt / (1 - (beta2 ** (i + 1)))
    vtt_sqrt = np.array([np.sqrt(vtt[0]), np.sqrt(vtt[1])])
    B0 -= r * mtt / (vtt_sqrt + eps)  # 更新梯度

# 绘制散点图
plt.subplot(221)
plt.scatter(x, y, c='b')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.grid()
# 绘制拟合曲线
plt.subplot(222)
plt.scatter(x, y, c='b')
plt.xlabel('petal length')
plt.ylabel('petal width')
y_ = np.dot(Data, B0)
plt.plot(x, y_, c='r')
plt.grid()

plt.subplot(212)
x_axis = np.arange(1, 301)
plt.plot(x_axis, loss[0:300])
plt.show()
print('iter=%s,loss=%s' % (iter, loss[-1]))
print(B0)