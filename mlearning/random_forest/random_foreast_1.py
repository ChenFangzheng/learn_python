'''
Use Indian liver patient dataset to make a random forest for predicting whether a person is suffering from liver problem or not
https://tektrace.wordpress.com/2016/04/01/random-forest-implementation-in-pythonscikit-learn/
'''
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing

# os.chdir('')
AH_DATA = pd.read_csv('~/workspace/git/learn_python/mlearning/data/Indian Liver Patient Dataset (ILPD).csv')
# os.chdir('../random_forest')

DATA_CLEAN = AH_DATA.dropna()

le = preprocessing.LabelEncoder()
le.fit(DATA_CLEAN.Gender)
DATA_CLEAN.loc[:, 'Gender'] = le.transform(DATA_CLEAN.Gender)
predictors = DATA_CLEAN[[column for column in DATA_CLEAN.keys()[:-1]]]
targets = DATA_CLEAN.Target

pred_train, pred_test, tar_train, tar_test = train_test_split(
    predictors, targets, test_size=.4)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=25)
classifier = classifier.fit(pred_train, tar_train)
predictions = classifier.predict(pred_test)
print(sklearn.metrics.confusion_matrix(tar_test, predictions))
print(sklearn.metrics.accuracy_score(tar_test, predictions))

model = ExtraTreesClassifier()
model.fit(pred_train, tar_train)

print(model.feature_importances_)

trees = range(25)
accuracy = np.zeros(25)

for idx in range(len(trees)):
    classifier = RandomForestClassifier(n_estimators=idx + 1)
    classifier = classifier.fit(pred_train, tar_train)
    predictions = classifier.predict(pred_test)
    accuracy[idx] = sklearn.metrics.accuracy_score(tar_test, predictions)

plt.cla()
plt.plot(trees, accuracy)
plt.show()
