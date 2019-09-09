"""
计算数据集中的离群属性
"""

import math
import numpy as np

def comentropy(attribute):
    """
    计算某一属性的信息熵
    :param attribute:该属性 一维数组类型
    :return:信息熵 float类型
    """
    attri2accurrence = {}
    for a in attribute:
        if a in attri2accurrence: attri2accurrence[a] += 1
        else: attri2accurrence[a] = 1

    totalNum = len(attribute)

    res = 0.0
    for k in attri2accurrence.keys():
        p = attri2accurrence[k] / totalNum
        res -= p * math.log2(p)
    return res

def getOutlierAttri(X):
    """
    计算并返回离群属性
    :param X:数据集 numpy二维数组类型
    :return: 离群属性的index list类型
    """
    if not type(X) is np.dtype:
        X = np.array(X,dtype=float)

    attriNum = X.shape[1]
    comenOfAttri = []
    totalComen = 0.0
    for i in range(attriNum):
        c = comentropy(X[:,i])
        comenOfAttri.append(c)
        totalComen += c
    threshold = totalComen / attriNum

    res = []
    for i in range(len(comenOfAttri)):
        if comenOfAttri[i] >= threshold: res.append(i)

    return res

def getOmegas(X, alpha):
    m = len(X[0])
    outliers = getOutlierAttri(X)
    res = np.zeros(m, dtype=float)
    for ol in outliers:
        res[ol] = alpha
    return res

if __name__ == '__main__':
    test = [
        [1,2,3],
        [4,4,4],
        [1,4,7],
        [1,6,4]
    ]
    omegas = getOmegas(test, 1)
    print(omegas)

