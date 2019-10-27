import numpy as np
import matplotlib.pyplot as plt
from math import log

a = [['高', '本科', '是', '大于', '10', 'yes'], ['高', '研究生', '是', '小于', '9', 'yes'],
     ['中', '本科', '否', '小于', '9', 'no'], ['高', '本科', '否', '等于', '9', 'no'],
     ['低', '专科', '是', '等于', '9', 'no'],['高', '研究生', '是', '等于', '10', 'no'],
     ['低', '专科', '否', '等于', '9', 'no'],['中', '专科', '是', '大于', '9', 'yes']]
a = np.array(a)
b=['身高','学历','是否公务员','是否大于平均工资','每年出差数','是否相亲']
b=np.array(b)

# 计算香农熵
def calcshannonEnt(dataSet):
    numEntries = len(dataSet)  # 计算出样本个数
    labelCount = {}  # 构造一个字典用于存储各类样本个数
    for result in dataSet:
        result = result[-1]
        if result in labelCount.keys():
            labelCount[result] += 1
        else:
            labelCount[result] = 1
    shannonEnt = 0.0
    for key in labelCount:
        pi = labelCount[key] / numEntries
        shannonEnt -= pi * log(pi, 2)
    return shannonEnt

# 数据的划分
def splitData(dataSet, axis, value):
    rownum = len(dataSet[:, 0])
    b = []
    for i in range(rownum):
        if dataSet[i, axis] == value:
            b.append(i)
    return np.delete(dataSet[b, :], axis, 1)
# 将元素加入字典，计算出每个值的出现次数
def labelCount(data):
    labelCount = {}
    for value in data:
        if value not in labelCount.keys():
            labelCount[value] = 1
        else:
            labelCount[value] = labelCount[value] + 1
    return labelCount

# 选中最好的划分
def chooseBestsplit(dataSet):
    datashanon = calcshannonEnt(dataSet)
    columnNum = len(dataSet[0, :]) - 1
    bestShanon = 0.0
    rowNum = len(dataSet[:, 0])
    for i in range(columnNum):
        splitShanon = 0.0
        shannonValues = labelCount(dataSet[:, i])
        tmp = np.unique(dataSet[:, i])
        for value in tmp:
            splitShanon += (shannonValues[value] / rowNum) * calcshannonEnt(splitData(dataSet, i, value))
        infogain = datashanon - splitShanon
        if infogain >= bestShanon:
            bestShanon = infogain
            bestSplit = i
    return bestSplit



# 递归构造决策树
def createTree(dataSet,label):
    if len(np.unique(dataSet[:, -1])) == 1:
        return dataSet[0, -1]
    if len(dataSet[0, :]) == 2:
        dictValue = labelCount(dataSet[:, -1])
        return max(dictValue, key=dictValue.get)
    bestFeat=chooseBestsplit(dataSet)
    bestFeatLabel=label[bestFeat]
    tmp = np.unique(dataSet[:, bestFeat])
    label=np.delete(label,bestFeat)
    myTree={bestFeatLabel:{}}
    tmp = np.unique(dataSet[:, bestFeat])
    for value in tmp:
        labelSub=label[:]
        myTree[bestFeatLabel][value]=createTree(splitData(dataSet,bestFeat,value),labelSub)
    return myTree
print(createTree(a,b))
