import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import fowlkes_mallows_score, calinski_harabasz_score, silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

matplotlib.rcParams['axes.unicode_minus'] = False
# 读取数据集
wine = pd.read_csv("wine.csv")
wine_quality = pd.read_csv("wine_quality.csv", sep=';')

# 将数据集的数据和标签拆分开
wine_labels = wine.iloc[:, 0]
wine = wine.iloc[:, 1:]
wine_data = wine.values
wine_quality_labels = wine_quality.iloc[:, -1]
wine_quality = wine_quality.iloc[:, :-1]
wine_quality_data = wine_quality.values
# print("\nwine数据集的数据：")
# print(wine_data)
# print("\nwine_quality数据集的数据：")
# print(wine_quality_data)
# print("\nwine数据集的标签")
# print(wine_labels)
# print("\nwine_quality数据集的标签")
# print(wine_quality_labels)

# 将数据集划分为训练集和测试集 测试集比例为10%  (test_size表示测试集对数据集的占比  random_state确保实验结果可复现)
wine_data_train, wine_data_test, wine_labels_train, wine_labels_test = train_test_split(wine_data, wine_labels,
                                                                                        test_size=0.1, random_state=42)
wine_quality_data_train, wine_quality_data_test, wine_quality_labels_train, wine_quality_labels_test = train_test_split(
    wine_quality_data, wine_quality_labels, test_size=0.1, random_state=42)
# print("\nwine数据集的训练集：")
# print(wine_data_train)
# print("\nwine_quality数据集的训练集：")
# print(wine_quality_data_train)

# 标准化数据集 对wine数据集采用均值标准化
wine_data_train_df = pd.DataFrame(wine_data_train)
scaler = StandardScaler()
wine_train_standardized = scaler.fit_transform(wine_data_train_df)
wine_data_test_df = pd.DataFrame(wine_data_test)
scaler = StandardScaler()
wine_test_standardized = scaler.fit_transform(wine_data_test_df)
# print("\nwine数据训练集的均值标准化结果：")
# print(wine_train_standardized)
# print("\nwine数据测试集的均值标准化结果：")
# print(wine_test_standardized)

# 标准化数据集 对wine_quality数据集采用均值标准化
wine_quality_data_test_df = pd.DataFrame(wine_quality_data_test)
scaler = StandardScaler()
wine_quality_test_standardized = scaler.fit_transform(wine_quality_data_test_df)
wine_quality_data_train_df = pd.DataFrame(wine_quality_data_train)
scaler = StandardScaler()
wine_quality_train_standardized = scaler.fit_transform(wine_quality_data_train_df)
# print("\nwine_quality数据训练集的均值标准化结果：")
# print(wine_quality_train_standardized)
# print("\nwine_quality数据测试集的均值标准化结果：")
# print(wine_quality_test_standardized)

# PCA降维 n_components表示保留前五个方差最大的特征向量
# 对wine数据集进行PCA降维
pca = PCA(n_components=2).fit(wine_train_standardized)
wine_trainPCA = pca.transform(wine_train_standardized)
wine_testPCA = pca.transform(wine_test_standardized)
# print("\nwine数据训练集的PCA降维结果：")
# print(wine_trainPCA)
# print("\nwine数据测试集的PCA降维结果：")
# print(wine_testPCA)

# 对wine_quality数据集进行PCA降维
pca = PCA(n_components=2).fit(wine_quality_train_standardized)
wine_quality_trainPCA = pca.transform(wine_quality_train_standardized)
wine_quality_testPCA = pca.transform(wine_quality_test_standardized)
# print("\nwine_quality数据训练集的PCA降维结果：")
# print(wine_quality_trainPCA)
# print("\nwine_quality数据测试集的PCA降维结果：")
# print(wine_quality_testPCA)

# K-means聚类分析
kmeans = KMeans(n_clusters=3, random_state=1, n_init=10).fit(wine_trainPCA)
print("\nKMeans聚类结果：")
kmeans.labels_ += 1
print(kmeans.labels_)
# 聚类结果可视化：绘制散点图(黑色表示噪点)
plt.figure(figsize=(8, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.scatter(wine_trainPCA[:, 0], wine_trainPCA[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker='x')

plt.show()

# Calinski-Harabasz评价模型
calinski_harabasz_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=1, n_init=10).fit(wine_trainPCA)
    score = calinski_harabasz_score(wine_trainPCA, kmeans.labels_)
    calinski_harabasz_scores.append(score)
    print("簇数为%d时的Calinski-Harabasz指数为：%f" % (i, score))

# 输出最优的簇数
best_k = calinski_harabasz_scores.index(max(calinski_harabasz_scores)) + 2
print("Calinski-Harabasz评价最优的簇数为：%d" % best_k)

# 轮廓系数评价模型
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=1, n_init=10).fit(wine_trainPCA)
    score = silhouette_score(wine_trainPCA, kmeans.labels_)
    silhouette_scores.append(score)

# 绘制轮廓系数曲线
plt.figure(figsize=(8, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(range(2, 11), silhouette_scores)
plt.xlabel("簇数")
plt.ylabel("轮廓系数")
plt.show()

# DBSCAN算法
# FMI评价模型
for i in range(500, 600, 1):
    dbscan = DBSCAN(eps=i * 0.001, min_samples=4).fit(wine_trainPCA)
    score = fowlkes_mallows_score(wine_labels_train, dbscan.labels_)
    print('eps为%.3f类FMI评价分值为：%f' % (i * 0.001, score))

# Calinski-Harabasz评价模型
calinski_harabasz_scores = []
for i in range(500, 600, 1):
    dbscan = DBSCAN(eps=i * 0.001, min_samples=4).fit(wine_trainPCA)
    score = calinski_harabasz_score(wine_trainPCA, dbscan.labels_)
    calinski_harabasz_scores.append(score)
    print("eps为%.3f时的Calinski-Harabasz指数为：%f" % (i * 0.001, score))

# 输出最优的簇数
best_k = calinski_harabasz_scores.index(max(calinski_harabasz_scores)) + 500
print("Calinski-Harabasz评价最优的eps为：0.%d" % best_k)

dbscan = DBSCAN(eps=0.571, min_samples=4).fit(wine_trainPCA)
print("dbscan:")
print(dbscan)
print("dbscan.labels_:")
print(dbscan.labels_)
dbscan.labels_ += 1

print(dbscan.labels_)
df4 = pd.DataFrame(wine_trainPCA)
plt.figure(figsize=(9, 6))
unique_labels = set(dbscan.labels_)  # 获取所有唯一的标签
colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))  # 为每个聚类生成不同的颜色

for label, col in zip(unique_labels, colors):
    if label == -1:  # 噪声点
        col = 'k'  # 黑色
    class_member_mask = (dbscan.labels_ == label)
    xy = df4[class_member_mask]
    plt.scatter(xy.iloc[:, 0], xy.iloc[:, 1], c=col, label=f'Cluster {label + 1}', alpha=0.5)

plt.legend()
plt.title('DBSCAN Clustering')
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, \
    mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

print('K-means模型评价结果：')
print('wine数据K-means模型的平均绝对误差为：',
      mean_absolute_error(wine_labels_train, kmeans.labels_))
print('wine数据K-means模型的均方误差为：',
      mean_squared_error(wine_labels_train, kmeans.labels_))
print('wine数据K-means模型的中值绝对误差为：',
      median_absolute_error(wine_labels_train, kmeans.labels_))
print('wine数据K-means模型的可解释方差值为：',
      explained_variance_score(wine_labels_train, kmeans.labels_))
print('wine数据K-means模型的R方值为：',
      r2_score(wine_labels_train, kmeans.labels_))

print('DBSCAN模型评价结果：')
from sklearn.metrics import explained_variance_score, \
    mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

print('wine数据DBSCAN树模型的平均绝对误差为：',
      mean_absolute_error(wine_labels_train, dbscan.labels_))
print('wine数据DBSCAN树模型的均方误差为：',
      mean_squared_error(wine_labels_train, dbscan.labels_))
print('wine数据DBSCAN树模型的中值绝对误差为：',
      median_absolute_error(wine_labels_train, dbscan.labels_))
print('wine数据DBSCAN树模型的可解释方差值为：',
      explained_variance_score(wine_labels_train, dbscan.labels_))
print('wine数据DBSCAN树模型的R方值为：',
      r2_score(wine_labels_train, dbscan.labels_))
