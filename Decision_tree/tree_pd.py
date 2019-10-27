import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from math import log
import  time
start = time.clock()
daset = pd.read_csv("data/test.csv")


# 计算香农熵，dataSet 是数据集，reslutLabel 是结果标签
def calcshannonEnt(dataSet, reslutLabel):
    numEntries = dataSet.shape[0]  # 计算出行数
    dataSet = dataSet[reslutLabel]
    labelCount = {}  # 构造一个字典用于存储各类样本个数
    for result in dataSet:
        if result in labelCount.keys():
            labelCount[result] += 1
        else:
            labelCount[result] = 1
    shannonEnt = 0.0
    for key in labelCount:  # 计算香农熵
        pi = labelCount[key] / numEntries
        shannonEnt -= pi * log(pi, 2)
    return shannonEnt


# 数据的划分 dataSet 数据集， label 需要划分的列标签 ，value 需要划分的值
def splitData(dataSet, label, value):
    dataSet = dataSet[dataSet[label].isin([value])]  # 根据传入的标签列和特征值进行划分
    return pd.DataFrame.drop(dataSet, label, axis=1)  # 去除传入的标签所在的列


# 选中最好的划分
def chooseBestsplit(dataSet, reslutLabel):
    datashanon = calcshannonEnt(dataSet, reslutLabel)
    columnNum = dataSet.shape[0]
    bestShanon = 0.0
    for label in dataSet.drop(reslutLabel, axis=1):  # 遍历除了结果项的所有标签
        labelKind = set(dataSet[label])  # set() 可以去除重复项
        splitShnnon = 0.0
        for value in labelKind:  # 遍历每一个特征 累加每个特征的香农熵
            splitShnnon += (sum(dataSet[label] == value) / columnNum) * calcshannonEnt(splitData(dataSet, label, value),
                                                                                       reslutLabel)
        infogain = datashanon - splitShnnon  # 得到信息增益值
        if infogain >= bestShanon:  # 选出最大的信息增益的分裂项
            bestShanon = infogain
            bestSplit = label
    return bestSplit


# 递归构造决策树
def createTree(dataSet, reslutLabel):
    if len(set(dataSet[reslutLabel])) == 1:  # 判断结果项是否唯一 如果唯一则返回一个结果
        return set(dataSet[reslutLabel]).pop()
    if dataSet.shape[1] == 2:  # 如果包括结果项还有两列，则取结果列中出现次数最多的值
        return Counter(dataSet[reslutLabel]).most_common()[0][0]  # Counter可以统计各个数据出现的次数 most_common 选择前几的几个 然后直接选第一个
    bestFeat = chooseBestsplit(dataSet, reslutLabel)
    myTree = {bestFeat: {}}  # 构建一个字典存储划分的数据
    for value in dataSet[bestFeat]:  # 递归构造决策树
        myTree[bestFeat][value] = createTree(splitData(dataSet, bestFeat, value), reslutLabel)
    return myTree


print(createTree(daset,"是否相亲"))
elapsed = (time.clock() - start)
print("Time used:",elapsed)