import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('15points.csv', header=None, names=['x1', 'x2', 'y'])
data_x = data.iloc[:, :2].values
data_y = data['y'].values.reshape(-1, 1)
x = np.linspace(0, 12, 200)
y = np.linspace(0, 8, 200)


# 定义欧式距离
def euclidean_distance(d1, d2):
    dist = np.sqrt(np.sum(np.power((d1 - d2), 2)))
    return dist

# 定义KNN
def KNN(x1, y1):
    coord = np.array([x1, y1])
    arr_dist = np.zeros((data_x.shape[0], 2))
    for i in range(data_x.shape[0]):
        dist = euclidean_distance(coord, data_x[i])
        arr_dist[i, :] = dist, data_y[i]
    arr_dist_df = pd.DataFrame(data=arr_dist, columns=['distance', 'target'])
    arr_sort = arr_dist_df.sort_values(by='distance')['target'].head(1).value_counts()
    sort_value = arr_sort.index[0]
    return sort_value


# 计算平面内每个点的最近邻，将每个点画出来，形成决策边界
for elementx in x:
    for elementy in y:
        sort_value = KNN(elementx, elementy)
        if sort_value == 1:
            plt.scatter(elementx, elementy, s=50, color='r')
        else:
            plt.scatter(elementx, elementy, s=50, color='b')

plt.scatter(data_x[:, 0:1][data_y == 1], data_x[:, 1:][data_y == 1], s=80, marker='o', color='black')
plt.scatter(data_x[:, 0:1][data_y == -1], data_x[:, 1:][data_y == -1], s=80, marker='x', color='black')
plt.show()
