#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
LARS测试代码

这里计算了LARS算法在diabetes数据集上，其正则化参数的路径

最终结果图中的每一个颜色代表参数向量中不同的特征
'''

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import datasets

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
print(X.shape)

alphas, _, coefs = linear_model.lars_path(X, y, method='lasso', verbose=True)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyles='dashed')
plt.xlabel('|coef|/max|coef|')
plt.ylabel("Lasso path")
plt.axis('tight')
plt.show()
