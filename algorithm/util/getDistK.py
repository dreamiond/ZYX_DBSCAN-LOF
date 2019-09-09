"""
计算某一条数据的第k距离
"""
from algorithm.util.dist import dist
from numpy.linalg.linalg import norm

def getDistK(X, i, k, omegas):
    """
    计算第i条数据的第k距离(有权重版本)
    :param X: 总数据集 numpy二维数组类型
    :param i: 第i条数据 int类型
    :param k: 第k距离 int类型
    :param omegas: 计算距离时使用到的权重
    :return: 第i条数据的第k距离 float类型
    """
    distances = []
    for vec in X:
        distances.append(dist(vec, X[i], omegas))
    distances.sort()
    return distances[k]

def getDistKWithoutOmegas(X, i, k):
    """
    计算第i条数据的第k距离(无权重版本)
    :param X: 总数据集 numpy二维数组类型
    :param i: 第i条数据 int类型
    :param k: 第k距离 int类型
    :return: 第i条数据的第k距离 float类型
    """
    distances = []
    for vec in X:
        distances.append(norm(vec - X[i]))
    distances.sort()
    return distances[k]

if __name__ == "__main__":
    test = [
        [1,2,3],
        [4,4,4],
        [1,4,7],
        [1,6,4]
    ]
    om = [0.0, 1.0, 1.0]
    print(getDistK(test,0,2,om))