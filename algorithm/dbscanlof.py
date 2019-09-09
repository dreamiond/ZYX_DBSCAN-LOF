import numpy as np
from algorithm.util.getOmegas import getOmegas
from algorithm.util.getDistK import getDistK,getDistKWithoutOmegas
from algorithm.util.dist import dist
from sklearn.neighbors.lof import LocalOutlierFactor

alpha = 1.0
k1 = 4
k2 = 4
k2UseOmegas = False
rho = 0.3 # 在每个簇中，选取k邻域密度较小的一部分的数据对象
nNoise = 50 # 	将D'中全部数据对象的LOF值从大到小排序，选择前n_noise个数据对象，得到数据集N_2
filePath = 'KDDTrain_nor_1000.csv'

def pyramid(centroid, radius, X, omegas, ignoreIdx):
    """
    对于一个核心对象centroid，以distk(centroid)为半径，找出所有由核心对象密度可达的点，生成簇
    :param centroid: 核心对象 int类型
    :param radius: 半径 float类型
    :param X: 原始数据集 numpy二维数组类型
    :param omegas: 计算向量距离时使用的权重 一位float数组类型
    :param ignoreIdx: 忽略的点（已经被划进其它簇的点） 一维int数组类型
    :return: 生成的簇 一维int数组类型
    """
    res = [centroid]
    n = X.shape[0]
    changed = True
    while changed:
        changed = False
        for point in res:
            for i in range(n):
                if (i not in res) and (i not in ignoreIdx) and dist(X[point], X[i], omegas) <= radius:
                    res.append(i)
                    changed = True
    return res

def notInClusters(point, clusters):
    if len(clusters) == 0: return True
    for vec in clusters:
        if point in vec: return False
    return True

def clusters2IngnorePoinst(clusters):
    if len(clusters) == 0: return []
    res = []
    for vec in clusters:
        for num in vec:
            res.append(num)
    return res

def sortClusterByDistK(vec, distKs):
    """
    将向量按k邻域密度从小到大排列
    :param vec: 要排序的向量 一维float数组类型
    :param distKs: k邻域密度 一维float数组类型
    :return: 排序好的向量
    """
    n = len(vec)
    for i in range(n):
        for j in range(i+1, n):
            if distKs[vec[i]] > distKs[vec[j]]:
                tmp = vec[i]
                vec[i] = vec[j]
                vec[j] = tmp
    return vec

def sortDByLofs(Didx, lofs):
    """
    将D中全部数据对象的LOF值从大到小排序
    :param Didx: D,元素为对象在X中的index
    :param lofs: lof值
    :return: 排序后的Didx
    """
    n = len(Didx)
    for i in range(n):
        for j in range(i+1, n):
            if lofs[i] < lofs[j]:
                tmp = lofs[i]
                lofs[i] = lofs[j]
                lofs[j] = tmp
                tmp = Didx[i]
                Didx[i] = Didx[j]
                Didx[j] = tmp
    return Didx

def dbscanlof(X):
    if not type(X) is np.dtype:
        X = np.array(X)

    n, m = X.shape
    omegas = getOmegas(X, alpha)
    distKs = []
    meanDistK = 0.0
    for i in range(n):
        d = getDistK(X, i, k1, omegas)
        distKs.append(d)
        meanDistK += d
    meanDistK /= n

    corePoints = []
    for i in range(n):
        if distKs[i] < meanDistK: corePoints.append(i)

    clusters = []
    for point in corePoints:
        if notInClusters(point,clusters):
            ignorePoints =clusters2IngnorePoinst(clusters)
            clust = pyramid(point, distKs[point], X, omegas, ignorePoints)
            if len(clust) != 1: clusters.append(clust)


    clusterPoints = []
    for vec in clusters:
        for point in vec:
            clusterPoints.append(point)
    noisePoints = []
    for i in range(n):
        if not i in clusterPoints:
            noisePoints.append(i)

    # 为各个簇中的各个数据对象计算k邻域密度
    distKs2 = []
    if k2UseOmegas:
        distKs2 = distKs
    else:
        for i in range(n):
            distKs2.append(getDistKWithoutOmegas(X, i, k2))

    # 将每个簇中的各个对象的按k邻域密度从小到大排列
    sortedClusters = []
    for vec in clusters:
        sortedClusters.append(sortClusterByDistK(vec, distKs2))

    # 在每个簇中，选取k邻域密度较小的一部分的数据对象
    Didx = noisePoints.copy()
    for vec in sortedClusters:
        for i in range(int(rho * len(vec))):
            Didx.append(vec[i])
    D = []
    for i in Didx:
        D.append(X[i])
    # 遍历D'中的每一个数据对象，计算每个数据对象的LOF值
    clf = LocalOutlierFactor(n_neighbors=k2, algorithm='auto', contamination=0.1, n_jobs=-1, p=2)
    clf.fit(D)
    lofs = clf._decision_function(D)
    sortedDidx = sortDByLofs(Didx, lofs)
    n2 = sortedDidx[:nNoise]

    res = []
    for p in noisePoints:
        if p in n2:
            res.append(p)
    return res

if __name__ == '__main__':
    data = np.genfromtxt(filePath, dtype=float, delimiter=',')
    print(dbscanlof(data))