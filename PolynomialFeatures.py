# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 09:46:41 2018

@author: Administrator
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

os.chdir('C:/Users/Administrator/Desktop/jpynb/机器学习')
df = pd.read_csv('./data/housing.csv')
df.head()

# 载入数据
X = df[['LSTAT']].values
y = df['MEDV'].values

# 添加模型
pr = LinearRegression()
lr = LinearRegression()
# degree多项式的阶数
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

# 用来做预测的数据，np.newaxis规范矩阵计算
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

# 训练简单线性模型
lr.fit(X, y)
y_lin_fit = lr.predict(X_fit)

# 训练多项式模型
pr.fit(X_quad, y)
# fit_transform对数据预处理，先调用fit()，后调用transform()
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

# 画出结果图
plt.scatter(X, y, label='training points')
plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit, label='quadratic fit')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('C:/Users/Administrator/Desktop/2.png', dpi=300)
