#!/usr/bin/python
#-*- coding:utf-8-*-
'''
Lasso 回归应用于稀疏信号(坐标下降)
refer: http://blog.csdn.net/daunxx/article/details/51596877
'''
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

np.random.seed(int(time.time()))
N_SAMPLES, N_FEATURES = 50, 200
X = np.random.randn(N_SAMPLES, N_FEATURES)
COEF = 3 * np.random.randn(N_FEATURES)
INDEX = np.arange(N_FEATURES)
np.random.shuffle(INDEX)
COEF[INDEX[10:]] = 0
y = np.dot(X, COEF)
y += 0.01 * np.random.normal((N_SAMPLES,))

N_SAMPLES = X.shape[0]
X_train, y_train = X[:int(N_SAMPLES / 2)], y[:int(N_SAMPLES / 2)]
X_test, y_test = X[int(N_SAMPLES / 2):], y[int(N_SAMPLES / 2):]

alpha = 0.1
lasso = Lasso(max_iter=10000, alpha=alpha)
y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)

# 这里是R2可决系数（coefficient of determination）
# 回归平方和（RSS）在总变差（TSS）中所占的比重称为可决系数
# 可决系数可以作为综合度量回归模型对样本观测值拟合优度的度量指标。
# 可决系数越大，说明在总变差中由模型作出了解释的部分占的比重越大，模型拟合优度越好。
# 反之可决系数小，说明模型对样本观测值的拟合程度越差。
# R2可决系数最好的效果是1。
R2_SCORE_LASSO = r2_score(y_test, y_pred_lasso)
print("测试集上的R2可决系数 : %f" % R2_SCORE_LASSO)

plt.plot(lasso.coef_, label='Lasso coefficients')
plt.plot(COEF, '--', label='original coefficients')
plt.legend(loc='best')
plt.show()

