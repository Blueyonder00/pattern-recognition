import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 定义softmax函数
def f_softmax(z):
    y = np.exp(z)
    return y / np.sum(y, axis=0)
# 定义交叉熵损失函数
def cross_entropy(y,p):
    L1=-np.multiply(y,np.log(p))
    L=np.sum(L1)
    return L

iris_df = datasets.load_iris()
x_ = iris_df.data
y = iris_df.target
x = np.hstack((x_, np.ones((x_.shape[0], 1))))  # 将x右侧加上一列1，对应权值b
print(x_)


y_onehot = np.eye(3)[y]  # 将输出值转换为one-hot形式 y_onehot(150,3)

W=np.random.randn(3,5) # 初始化权值
r=0.01 # 学习率
loss=[]
for i in range(10000):
    p = f_softmax(W.dot(x.T))  # p(3,150)
    deltaW = (p - y_onehot.T).dot(x) # deltaW(3,5)
    W-=r*deltaW
    loss.append(cross_entropy(y_onehot,p.T))

x_axis = np.arange(1,301)
plt.plot(x_axis,loss[0:300])
plt.show()


