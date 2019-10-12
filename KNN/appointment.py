import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import operator

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 将文本记录转换为 Numpy 的解析程序
def filematrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()  # 得到文本所有行,放入数组中
    numberOfLines = len(arrayOLines)  # 得到行数
    returnMat = np.zeros((numberOfLines, 3))  # 构造一个全 0 矩阵
    classLabelVectirt = []  # 用于存储标签
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 去除首尾空格或换行符
        listFromLine = line.split('\t')  # 以空格进行分割 存入数组
        returnMat[index, :] = listFromLine[0:3]  # 将每一行数据依次加入矩阵中
        classLabelVectirt.append(listFromLine[-1])  # 将最后一列加入标签
        index += 1
    return returnMat, classLabelVectirt


# 画散点图
def DrawData(dataingLables):
    fig = plt.figure()  # 创建一个画布
    ax = fig.add_subplot(311)  # 分割为三块,在第一块位置
    plt.xlabel("飞行时间")
    plt.ylabel("每周冰淇淋公升数")
    bx = fig.add_subplot(312)  # 分割为三块,在第二块位置
    plt.xlabel("飞行时间")
    plt.ylabel("游戏时间")
    cx = fig.add_subplot(313)  # 分割为三块,在第三块位置
    plt.xlabel("游戏时间")
    plt.ylabel("冰淇淋")
    LablesNum = dataingLables.copy()
    k = 0
    for labdata in dataingLables:
        if labdata == "didntLike":
            LablesNum[k] = 1
        elif labdata == "smallDoses":
            LablesNum[k] = 2
        else:
            LablesNum[k] = 3
        k += 1
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 2], c=LablesNum)  # 根据数据画出散点图
    bx.scatter(datingDataMat[:, 0], datingDataMat[:, 1], c=LablesNum)
    cx.scatter(datingDataMat[:, 1], datingDataMat[:, 2], c=LablesNum)
    plt.show()


# 对数据进行归一化处理
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 0 代表选取每一列的最小值，1 代表每一行
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(dataSet.shape)  # 构造一个全 0 矩阵，行列等于 dataSet 的行列
    m = dataSet.shape[0]  # 得到矩阵的行数
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # tile 可以看做复制，这里复制 m 行 1 列个 minVals 作为矩阵，然后进行矩阵的减法运算
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # 除以差值得到 0-1 数据
    return normDataSet, ranges, minVals


def classify(inX, dataSet, labels, k):  # inX 测试数据 dataSet 训练数据 labels 标签项
    dataSetSize = dataSet.shape[0]  # 行数
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 求差值
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 每一列分别相加
    distances = sqDistances ** 0.5  # 得到欧式距离
    sortedDistInicies = distances.argsort()  # 返回从大到小的顺序 如 [3,2.2,3.1,2.5] 返回 [1 3 2 0]
    classCount = {}  # 字典
    for i in range(k):
        voteIlabel = labels[sortedDistInicies[i]]  # 得到第 i 个值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 将值作为键 出现相同的则加 1 默认为0
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),
                                  reverse=True)  # 降序排序 获取第一个值  即是获得频率最高值
        return sortedClassCount[0][0]


datingDataMat, dataingLables = filematrix('data/datingTestSet.txt')
normDataSet, ranges, minVals = autoNorm(datingDataMat)
DrawData(dataingLables)
while True:
    a = float(input("请输入飞行时间"))
    b = float(input("请输入冰淇淋数"))
    c = float(input("请输入游戏时间"))
    d = np.array([a, b, c])
    inX = (d - minVals) / ranges
    returnFine = classify(inX, normDataSet, dataingLables, 20)
    print("这个男人你可能", returnFine)
