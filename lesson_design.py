# 课程设计名称
# 使用scikit-learn构建和评价xx模型
# 课程设计内容
# wine数据集和wine_quality数据集是两份和葡萄酒有关的数据集。
# wine数据集包含3种同起源的葡萄酒的记录，共178条。其中，每个特征对应葡萄酒的每种化学成分,并且都属于连续型数据。通过化学分析可以推断葡萄酒的起源。
# wine_quality数据集共有1599个观察值，11个输入特征和一个标签。其中，不同类的观察值数量不等，所有特征为连续型数据。通过酒的各类化学成分，预测该葡萄酒的评分。
# (1) 使用pandas库分别读取wine数据集和wine_quality数据集；将wine数据集和wine_quality数据集的数据和标签拆分开；将wine数据集和wine_quality数据集划分为训练集和测试集；标准化wine数据集和wine_quality数据集；对wine数据集和wine_quality数据集进行PCA降维。
# (2) 根据(1)的wine数据集处理的结果，采用2种不同的聚类算法分别构建聚类模型；然后，通过定量指标评价所构建的模型的优劣。
# (3) 根据(1)的wine数据集处理的结果，采用2种不同的分类算法分别构建分类模型；然后，通过定量指标评价所构建的模型的优劣。
# (4) 根据(1)的wine_quality数据集处理的结果，采用2种不同的回归算法分别构建回归模型；然后，通过定量指标评价所构建的模型的优劣。
# 课程设计要求
# (1) 分组，每组3人。以小组为单位完成以下任务。
# (2) 编程实现“课程设计内容”中的(2)、(3)或(4)，任选1题；（实现思路和步骤可参见参考资料(2)的P196和P197页）
# (3) 撰写课程设计报告；（需要包含所采用算法的简介以及评价指标的简介；代码可作为附件放在正文之后）
# (4) 讲解展示，每组10分钟（每个人都要讲，时间和内容自行分配）。要求讲解清晰，使现场的同学们能够听懂，然后由评议小组实时评分。
# 参考资料
# (1) 课件的第11章PPT；
# (2)《Python数据分析与应用》，作者：黄红梅、张良均：人民邮电出版社，第6章 使用scikit-learn构建模型

# 设置环境变量UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
import os
os.environ['OMP_NUM_THREADS'] = '1'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

import pandas as pd


# (1) 使用pandas库分别读取wine数据集和wine_quality数据集；将wine数据集和wine_quality数据集的数据和标签拆分开；将wine数据集和wine_quality数据集划分为训练集和测试集；
# 标准化wine数据集和wine_quality数据集；对wine数据集和wine_quality数据集进行PCA降维。
# ……………………BEGIN…………………… #
# 读取数据集
wine = pd.read_csv("wine.csv")
wine_quality = pd.read_csv("wine_quality.csv", sep=';')

# 将数据集的数据和标签拆分开
wine = wine.iloc[:, 1:]
wine_labels = wine.iloc[:, 0]
wine_data = wine.values
wine_quality = wine_quality.iloc[:, :-1]
wine_quality_labels = wine_quality.iloc[:, -1]
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
print("\nwine数据训练集的均值标准化结果：")
print(wine_train_standardized)
print("\nwine数据测试集的均值标准化结果：")
print(wine_test_standardized)

# 标准化数据集 对wine_quality数据集采用均值标准化
wine_quality_data_test_df = pd.DataFrame(wine_quality_data_test)
scaler = StandardScaler()
wine_quality_test_standardized = scaler.fit_transform(wine_quality_data_test_df)
wine_quality_data_train_df = pd.DataFrame(wine_quality_data_train)
scaler = StandardScaler()
wine_quality_train_standardized = scaler.fit_transform(wine_quality_data_train_df)
print("\nwine_quality数据训练集的均值标准化结果：")
print(wine_quality_train_standardized)
print("\nwine_quality数据测试集的均值标准化结果：")
print(wine_quality_test_standardized)

# PCA降维 n_components表示保留前五个方差最大的特征向量
# 对wine数据集进行PCA降维
pca = PCA(n_components=5).fit(wine_train_standardized)
wine_trainPCA = pca.transform(wine_train_standardized)
wine_testPCA = pca.transform(wine_test_standardized)
print("\nwine数据训练集的PCA降维结果：")
print(wine_trainPCA)
print("\nwine数据测试集的PCA降维结果：")
print(wine_testPCA)
# 对wine_quality数据集进行PCA降维
pca = PCA(n_components=5).fit(wine_quality_train_standardized)
wine_quality_trainPCA = pca.transform(wine_quality_train_standardized)
wine_quality_testPCA = pca.transform(wine_quality_test_standardized)
print("\nwine_quality数据训练集的PCA降维结果：")
print(wine_quality_trainPCA)
print("\nwine_quality数据测试集的PCA降维结果：")
print(wine_quality_testPCA)

# ……………………END…………………… #

# (2) 根据(1)的wine数据集处理的结果，采用2种不同的聚类算法分别构建聚类模型；然后，通过定量指标评价所构建的模型的优劣。
# ……………………BEGIN…………………… #

# KMeans聚类
kmeans = KMeans(n_clusters=3,random_state=1, n_init=10).fit(wine_trainPCA)
print("\nKMeans聚类结果：")
print(kmeans.labels_)
#聚类结果可视化：绘制散点图(黑色表示噪点)
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
    print("簇数为%d时的轮廓系数为：%f" % (i, score))

# 绘制轮廓系数曲线
plt.figure(figsize=(8, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(range(2, 11), silhouette_scores)
plt.xlabel("簇数")
plt.ylabel("轮廓系数")
plt.show()

#输出最优的簇数
best_k = silhouette_scores.index(max(silhouette_scores)) + 2
print("轮廓系数评价最优的簇数为：%d" % best_k)

# 层次聚类
agg = AgglomerativeClustering(n_clusters=3).fit(wine_trainPCA)

# ……………………END…………………… #


