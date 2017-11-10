#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
基于lasso的特征选择
这个功能一般和其他的分类器一起使用
或直接内置于其他分类器算中
"""

import numpy as np
import time

from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score

iris = load_iris()
X, y = iris.data, iris.target
print('生成矩阵尺寸：%d, %d' % X.shape)

inds = np.arange(X.shape[0])
np.random.shuffle(inds)

X_train = X[inds[:100]]
y_train = y[inds[:100]]
X_test = X[inds[100:]]
y_test = y[inds[100:]]

print("原始特征维度 %d" % X_train.shape[1])

lsvc = LinearSVC(C=0.01, penalty='l1', dual=False).fit(X_train, y_train)
print("原始特征，在测试集上的准确率： %f" % lsvc.score(X_test, y_test))
print("原始特征，在测试集上的R2可决系数： %f" % r2_score(lsvc.predict(X_test), y_test))

model = SelectFromModel(lsvc, prefit=True)

X_train_new = model.transform(X_train)
X_test_new = model.transform(X_test)

print("新特征的维度：%d" % X_train_new.shape[1])

lsvc_new = LinearSVC(C=0.01, penalty='l1', dual=False).fit(
    X_train_new, y_train)

print("新特征，在测试集上的准确率： %f" % lsvc_new.score(X_test_new, y_test))
print("新特征，在测试集上的R2可决系数： %f" % r2_score(lsvc_new.predict(X_test_new), y_test))
