from sklearn.cluster import DBSCAN
import numpy as np

filePath = '../../algorithm/KDDTrain_nor_1000.csv'

X = np.genfromtxt(filePath, dtype=float, delimiter=',')

# eps：两个点能被划分为同一类的最小距离
# min_samples：被视为簇需含有的最少点的个数
cluster = DBSCAN(eps=1, min_samples=5).fit_predict(X=X)
print(cluster)
