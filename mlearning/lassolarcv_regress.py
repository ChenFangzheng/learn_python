'''
    refer to: https://tektrace.wordpress.com/2016/04/09/lasso-regression-in-python-scikit-learn/
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLarsCV
from sklearn import preprocessing

os.chdir('./data/')

data = pd.read_csv("Indian Liver Patient Dataset (ILPD).csv")

data_clean = data.dropna()

le = preprocessing.LabelEncoder()

le.fit(data_clean.Gender)

data_clean.loc[:, 'Gender'] = le.transform(data_clean.Gender)

predvar = data_clean[['Age', 'Gender', 'Total Bilirubin', 'Direct Bilirubin',
                      'Alkphos Alkaline Phosphotase', 'Sgpt Alamine Aminotransferase',
                      'Sgot Aspartate Aminotransferase', 'Total Proteins', 'ALB Albumin', 'Albumin and Globulin Ratio']]

target = data_clean.Target

predictors = predvar.copy()

predictors.loc[:, 'Age'] = preprocessing.scale(
    predictors['Age'].astype('float64'))
predictors.loc[:, 'Gender'] = preprocessing.scale(
    predictors['Gender'].astype('float64'))
predictors.loc[:, 'Total Bilirubin'] = preprocessing.scale(
    predictors['Total Bilirubin'].astype('float64'))
predictors.loc[:, 'Alkphos Alkaline Phosphotase'] = preprocessing.scale(
    predictors['Alkphos Alkaline Phosphotase'].astype('float64'))
predictors.loc[:, 'Sgpt Alamine Aminotransferase'] = preprocessing.scale(
    predictors['Sgpt Alamine Aminotransferase'].astype('float64'))
predictors.loc[:, 'Sgot Aspartate Aminotransferase'] = preprocessing.scale(
    predictors['Sgot Aspartate Aminotransferase'].astype('float64'))
predictors.loc[:, 'Total Proteins'] = preprocessing.scale(
    predictors['Total Proteins']. astype('float64'))
predictors.loc[:, 'ALB Albumin'] = preprocessing.scale(
    predictors['ALB Albumin'] .astype('float64'))
predictors.loc[:, 'Albumin and Globulin Ratio'] = preprocessing.scale(
    predictors['Albumin and Globulin Ratio'].astype('float64'))

pre_train, pre_test, tar_train, tar_test, = train_test_split(
    predictors, target, test_size=.3, random_state=123)

model = LassoLarsCV(cv=10, precompute=False).fit(pre_train, tar_train)

print(list(zip(predictors.columns, model.coef_)))


from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pre_train))
test_error = mean_squared_error(tar_test, model.predict(pre_test))
print('training data MSE')
print(train_error)
print('test data MSE')
print(test_error)

rsquared_train = model.score(pre_train, tar_train)
rsquared_test = model.score(pre_test, tar_test)
print('training data R-square')
print(rsquared_train)
print('test data R-square')
print(rsquared_test)


m_log_alphascv = -np.log10(model.cv_alphas_)

plt.figure()
plt.plot(m_log_alphascv, model.mse_path_, ':')
plt.plot(m_log_alphascv, model.mse_path_.mean(axis=-1),
         'k', label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--',
            color='k', label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
plt.show()


m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='-',
            color='k', label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.legend()
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')
plt.show()
