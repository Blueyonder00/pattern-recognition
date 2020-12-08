import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def classify(train, i, Q, S):
    answer = np.ones((train.shape[0], 1))
    if S == 'less':
        answer[train[:, i] <= Q] = -1
    else:
        answer[train[:, i] > Q] = -1
    return answer


# 构建单层决策树分类函数
# 输入：x_train,y_train,D(权重)
def get_best_stump(x_train, y_train, D):
    m, n = x_train.shape  # 返回样本个数和特征数
    best_stump = {}  # 用字典来存储树桩信息
    bestclass = np.zeros((m, 1))  # 初始化分类结果为1
    minError = 1
    for i in range(n):  # 遍历每个特征
        x_max = np.max(x_train[:, i])
        x_min = np.min(x_train[:, i])
        step = (x_max - x_min) / m  # 计算步长
        for j in range(1, m):  # 遍历每个阈值
            for S in ['less', 'more']:
                Q = (x_min + j * step)  # 计算阈值
                answer = classify(x_train, i, Q, S)
                error = np.ones((m, 1))
                error[answer == y_train] = 0
                rate = D.dot(error)
                if rate[0] < minError:
                    minError = rate[0]
                    bestclass = answer.copy()
                    best_stump['特征列'] = i
                    best_stump['阈值'] = Q
                    best_stump['标志'] = S
    return best_stump, minError, bestclass


# D = (np.ones((train_x.shape[0], 1)) / train_x.shape[0]).T
# best_stump,minerror,beatcalss=get_best_stump(train_x,train_y,D)
# print(best_stump,minerror,beatcalss)

# 构建Adaboost函数
def adaboost_fun(x_train, y_train, iter=50):
    weakclass = []
    new_class = np.zeros((x_train.shape[0], 1))
    D = (np.ones((x_train.shape[0], 1)) / x_train.shape[0]).T  # 初始化权值
    for i in range(iter):  # 对每次迭代，返回树桩信息，分类误差率，最好的单层决策树
        stump, em, bestclass = get_best_stump(x_train, y_train, D)
        alpham = 1 / 2 * np.log((1 - em) / max(em,1e-16))  # 计算弱分类器的权值
        stump['权重'] = np.round(alpham, 4)  # 存储弱分类器的权值
        weakclass.append(stump)
        temp_exp = np.exp(np.multiply(-alpham * y_train, bestclass))  # 计算指数项
        zm = D.dot(temp_exp)  # 计算规范化因子
        D = np.multiply(D, temp_exp.T) / zm  # 更新权值
        new_class += alpham * bestclass  # 更新累计类别估计值
        error_num = np.multiply(np.sign(new_class) != y_train, np.ones((x_train.shape[0], 1)))
        acc_error = np.sum(error_num) / float(x_train.shape[0])
        if acc_error == 0:
            break
    return weakclass

# 构建预测函数，返回分类结果
def predictions(x_test,weak_class):
    class_result=np.zeros((x_test.shape[0],1))
    num = len(weak_class) # 计算弱分类器的个数
    for i in range(num): # 遍历所有分类器
        class_single=classify(x_test,
                              weak_class[i]['特征列'],
                              weak_class[i]['阈值'],
                              weak_class[i]['标志'])
        class_result +=weak_class[i]['权重']*class_single # 计算所有分类器累计输出的结果
        return np.sign(class_result) # 返回分类结果

# 计算预测结果正确率
def cal_score(result,y_test):
    right=np.zeros((y_test.shape[0],1))
    right[result==y_test]=1
    score=float(sum(right))/y_test.shape[0]
    return score

if __name__ == '__main__':

    # 读取15个点数据
    data = pd.read_csv('15points.csv', delimiter=',', header=None, names=['x1', 'x2', 'y'])
    x = data[['x1', 'x2']].values
    y = data['y'].values.reshape(-1, 1)

    # 划分测试集和训练集
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.6, random_state=1)

    weakclass = adaboost_fun(train_x, train_y,iter=50) # 找到弱分类器
    result=predictions(test_x,weakclass) # 进行预测
    score = cal_score(result,test_y)
    print("弱分类器为：{}".format(weakclass))
    print("预测结果为：{}，准确率为{}".format(result,score))
