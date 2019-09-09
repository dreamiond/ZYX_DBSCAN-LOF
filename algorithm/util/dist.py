"""
计算基于信息熵的加权距离dist(p,q)
"""

import math

def dist(p, q, omegas):
    """
    根据信息熵计算两向量的加权距离
    :param p: 向量1 一维数组类型
    :param q: 向量2 一维数组类型
    :param omegas: 权重 一维数组类型
    :return: 两向量的加权距离 float类型
    """
    n, m, g = len(p), len(q),len(omegas)
    assert n == m and m == g and n == g, "向量维数不一致！"

    res = 0.0
    for i in range(n):
        res += (1 + omegas[i]) * (p[i] - q[i]) * (p[i] - q[i])
    res = math.sqrt(res)
    return res

if __name__ == "__main__":
    a = [1,2,3]
    b = [4,5,6]
    w = [0.5,1,1]
    print(dist(a, b, w))