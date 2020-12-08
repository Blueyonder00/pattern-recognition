import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

iris_df = datasets.load_iris()
x=iris_df.data
y=iris_df.target
# 划分测试集和训练集，训练集：测试集=4：1
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=None)

train=np.hstack((x_train,y_train.reshape(-1,1)))
test=np.hstack((x_test,y_test.reshape(-1,1)))

# 定义欧式距离
def Euclidean_distance(d1, d2):
    dist = np.sqrt(np.sum(np.power((d1 - d2), 2)))
    return dist


# 定义KNN
def KNN(train, test, k):
    true_num = 0
    for i in range(test.shape[0]):
        arr_dist = np.zeros((train.shape[0], 2))  # 初始化一个数组用于存放一个测试集样本和所有训练集样本的距离，
                                                  # 第一列为距离，第二列为该训练样本的类别
        for j in range(train.shape[0]):
            dist = Euclidean_distance(test[i, :-1], train[j, :-1])  # 计算每一个测试样本和训练样本的距离
            arr_dist[j, :] = dist, train[j, -1]  # 存放距离和类别
        arr_dist_df = pd.DataFrame(data=arr_dist, columns=['distance', 'target'])
        sort_df = arr_dist_df.sort_values(by='distance')['target'].head(k).value_counts()  # 对距离进行排序，选出前k个并按类别进行统计
        sort_value = sort_df.index[0]   # 选出前k个中出现最多的类别作为测试样本的类别
        if sort_value == test[i, -1]:   # 和测试样本所属原类别进行比较，统计预测正确的个数
            true_num += 1
    score = true_num / 30.0   # 计算score
    return score,true_num

score_list=[]
odd=list(range(1,15))[::2]
for k in odd:  # 对k取值为1-14时进行预测
    score,true_num = KNN(train, test, k)
    score_list.append(score)
    print('k=%d,score=%f,true number=%d' % (k, score,true_num))

# 画图，对k在不同取值情况下的score
plt.plot(odd,score_list,color='r',linestyle='-',marker='o',markerfacecolor='b')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.xticks(range(1,15)[::2]) # 坐标轴显示奇数，和k对应
plt.show()

