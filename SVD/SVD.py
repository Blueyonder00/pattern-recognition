import numpy as np
from PIL import Image

oriImage = Image.open('library.JPG')  # 读取图像，jpg格式有RGB三个颜色通道
imgArray = np.array(oriImage)  # 3D张量，由三个维度构成，分别为高度信息，宽度信息和颜色通道信息

# 将数组中的每个颜色通道单独分离出来,注意将每个通道的数值除以255.0,便于计算，同时也转换为float
R = imgArray[:, :, 0] / 255.0
G = imgArray[:, :, 1] / 255.0
B = imgArray[:, :, 2] / 255.0


# 定义通道矩阵压缩的函数，channel为单个颜色通道矩阵，percent为保留奇异值的百分比
def img_compress(channel, percent):
    '''eig_value, eig_vector = np.linalg.eigh(np.dot(channel, channel.T))  # R*R_T的特征向量和特征值
    eig_value_index = np.argsort(eig_value)[::-1]  # 特征值降序排序，得到索引值
    eig_value = np.sort(eig_value)[::-1]  # 特征值降序排列
    sigma = np.sqrt(eig_value[:channel.shape[1]])  # 特征值的平方根即为奇异值
    U = eig_vector[:, eig_value_index]  # 将特征值对应的特征向量排列好
    eig_value_, eig_vector_ = np.linalg.eigh(np.dot(channel.T, channel))
    eig_value_in = np.argsort(eig_value_)[::-1]
    V_T = eig_vector_[:, eig_value_in].T'''
    U1, sigma1, V_T1= np.linalg.svd(channel)  # 对通道矩阵进行奇异值分解
    m = U.shape[0]
    n = V_T.shape[0]
    rechannel = np.zeros((m, n))  # 初始化全零矩阵
    for i in range(m):
        rechannel += sigma[i] * np.dot(U[:, i].reshape(-1, 1), V_T[i, :].reshape(1, -1))  # 取前k个奇异值重建通道矩阵
        if i / m > percent:
            rechannel[rechannel < 0] = 0
            rechannel[rechannel > 1] = 1  # 把处理后的值约束到0-1范围内
            break
    rechannel *= 255.0
    return np.rint(rechannel).astype('uint8')


for p in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
    reR = img_compress(R, p)
    reG = img_compress(G, p)
    reB = img_compress(B, p)
    reImage = np.stack((reR, reG, reB), 2)
    Image.fromarray(reImage).save('{}'.format(p)+"library.JPG")
