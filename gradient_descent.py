# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 10:07:04 2018

@author: Administrator
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

os.chdir('C:/Users/Administrator/Desktop/jpynb/机器学习')

# 读取数据
df = pd.read_csv('./data/housing.csv')
#head()前五行的数据
df.head()

sns.set(style='whitegrid', context='notebook')  # 设定样式，还原可用 sns.reset_orig()
#context绘制上下文参数 style轴样式参数
# MEDV 是目标变量，为了方便演示，只挑 4 个预测变量
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

# scatterplot matrix, 对角线上是变量分布的直方图，非对角线上是两个变量的散点图
sns.pairplot(df[cols], size=3)
#plt.show()
plt.tight_layout()
# 这是matplotlib的方法，紧凑显示图片，居中显示

# 用下面这行代码可以存储图片到硬盘中
plt.savefig('C:/Users/Administrator/Desktop/scatter1.png', dpi=300)
#dpi是像素

cm = np.corrcoef(df[cols].values.T)  # 计算相关系数
sns.set(font_scale=1.5)
#font_scale 单独的缩放因子可以独立缩放字体元素的大小

# 画相关系数矩阵的热点图
hm = sns.heatmap(cm,
        annot=True,    #annot为True时，在heatmap中每个方格写入数据
        square=True,   #设置热力图矩阵小块形状，默认值是False
        fmt='.2f',   #矩阵上标识数字的数据格式
        annot_kws={'size': 11},   #annot为True时，可设置各个参数，包括大小，颜色，加粗，斜体字等
        yticklabels=cols,   #坐标轴标签名
        xticklabels=cols)
plt.tight_layout()
plt.savefig('C:/Users/Administrator/Desktop/corr_mat.png', dpi=300)

sns.reset_orig()



class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta  # learning rate 学习速率
        self.n_iter = n_iter  # 迭代次数

    def fit(self, X, y):  # 训练函数
        # self.w_ = np.zeros(1, 1 + X.shape[1])
        self.coef_ = np.zeros(shape=(1, X.shape[1]))  # 代表被训练的系数，初始化为 0
        self.intercept_ = np.zeros(1)
        self.cost_ = []   # 用于保存损失的空list

        for i in range(self.n_iter):
            output = self.net_input(X)  # 计算预测的Y
            errors = y - output
            self.coef_ += self.eta * np.dot(errors.T, X)  # 根据更新规则更新系数，思考一下为什么不是减号？
            self.intercept_ += self.eta * errors.sum()  # 更新 bias，相当于x_0取常数1
            cost = (errors**2).sum() / 2.0     # 计算损失
            self.cost_.append(cost)  # 记录损失函数的值
        return self

    def net_input(self, X):   # 给定系数和X计算预测的Y
        return np.dot(X, self.coef_.T) + self.intercept_   # dot矩阵乘法

    def predict(self, X):
        return self.net_input(X)
    
# RM 作为 explanatory variable
# RM 住宅平均房间数目
# MEDV 业主自住房屋中值 （要预测的变量）
X = df[['RM']].values
y = df[['MEDV']].values


# 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
# 归一化（标准化
sc_x = StandardScaler()
sc_y = StandardScaler()
# fit_transform()先拟合数据，然后转化它将其转化为标准形式
# transform()的作用是通过找中心和缩放等实现标准化
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)

lr = LinearRegressionGD()
lr.fit(X_std, y_std)  # 喂入数据进行训练

# cost function
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.tight_layout()
plt.savefig('C:/Users/Administrator/Desktop/cost.png', dpi=300)

# 定义一个绘图函数用于展示
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='lightblue')
    plt.plot(X, model.predict(X), color='red', linewidth=2)    
    return None

lin_regplot(X_std, y_std, lr)
plt.xlabel('[RM](standardized)')
plt.ylabel('[MEDV](standardized)')
plt.tight_layout()
plt.savefig('C:/Users/Administrator/Desktop/predict.png', dpi=300)

# 预测 RM=10 时，房价为多少
num_rooms_std = sc_x.transform([[10.0]]) 
price_std = lr.predict(num_rooms_std)
print("预测房价为: %.3f" % sc_y.inverse_transform(price_std))
