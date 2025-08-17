from imblearn.under_sampling import CondensedNearestNeighbour, ClusterCentroids, OneSidedSelection, TomekLinks, NearMiss,\
    RandomUnderSampler, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SMOTENC
import numpy as np
from sklearn.cluster import KMeans


def UnderSampling_methods(X, y, method):
    if method == 'km_uns':
        X_resampled, y_resampled = Kmeans_UnderSampling_methods(X, y)

    elif method == 'cnn':
        cnn = CondensedNearestNeighbour(random_state=0)
        X_resampled, y_resampled = cnn.fit_resample(X, y)

    elif method == 'enn':
        enn = EditedNearestNeighbours()
        X_resampled, y_resampled = enn.fit_resample(X, y)

    elif method == 'cc':
        cc = ClusterCentroids()
        X_resampled, y_resampled = cc.fit_resample(X, y)

    elif method == 'oss':
        oss = OneSidedSelection(random_state=0)
        X_resampled, y_resampled = oss.fit_resample(X, y)

    elif method == 'tl':
        tl = TomekLinks()
        X_resampled, y_resampled = tl.fit_resample(X, y)

    elif method == 'nm1':
        nm1 = NearMiss(version=1)
        X_resampled, y_resampled = nm1.fit_resample(X, y)

    elif method == 'rus':
        rus = RandomUnderSampler(random_state=0)
        X_resampled, y_resampled = rus.fit_resample(X, y)

    else:
        print('无可调用名为 {} 的欠采样方法'.format(method))

    return X_resampled, y_resampled

def Kmeans_UnderSampling_methods(X, y):
    # 根据类别0和类别1的数据量来区分多数类和少数类
    if (y == 0).sum() > (y == 1).sum():
        majority_class = X[y == 0]  # 根据标签数组 y 中的值为0的索引，从特征数组 X 中选择对应的样本，构成多数类样本数组 majority_class
        minority_class = X[y == 1]

        # 使用K-means对多数类样本进行聚类，并选择少数类样本数量的簇数
        n_clusters = len(minority_class)
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(majority_class)

        # 获取每个簇的中心点
        cluster_centers = kmeans.cluster_centers_

        # 组合欠采样后的样本集
        undersampled_X = np.concatenate((cluster_centers, minority_class))
        undersampled_y = np.concatenate((np.zeros(n_clusters), np.ones(len(minority_class))))
    else:
        majority_class = X[y == 1]
        minority_class = X[y == 0]

        # 使用K-means对多数类样本进行聚类，并选择少数类样本数量的簇数
        n_clusters = len(minority_class)
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(majority_class)

        # 获取每个簇的中心点
        cluster_centers = kmeans.cluster_centers_

        # 组合欠采样后的样本集
        undersampled_X = np.concatenate((cluster_centers, minority_class))
        undersampled_y = np.concatenate((np.ones(n_clusters), np.zeros(len(minority_class))))

    return undersampled_X, undersampled_y

def OverSampling_methods(X, y, method):
    # smote, borderline-smote, ada-smote
    if method == 'SMOTE':
        # X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        X_resampled, y_resampled = SMOTE(k_neighbors=1).fit_resample(X, y)
    elif method == 'ADASYN':
        X_resampled, y_resampled = ADASYN().fit_resample(X, y)  # n_neighbors=2
    elif method == 'BorderlineSMOTE':
        X_resampled, y_resampled = BorderlineSMOTE(k_neighbors=2).fit_resample(X, y)  # k_neighbors=2
    # elif method == 'SMOTENC':
    #     X_resampled, y_resampled = smote_nc.fit_resample(X, y)
    return X_resampled, y_resampled






