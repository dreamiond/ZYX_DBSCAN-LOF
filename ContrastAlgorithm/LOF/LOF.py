from sklearn.neighbors import LocalOutlierFactor
import numpy as np

filePath = '../../algorithm/KDDTrain_nor_1000.csv'

X = np.genfromtxt(filePath, dtype=float, delimiter=',')

# eps：两个点能被划分为同一类的最小距离
# min_samples：被视为簇需含有的最少点的个数
cluster = LocalOutlierFactor(n_neighbors=20, contamination=0.1).fit_predict(X=X)
print(cluster)