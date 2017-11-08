import time
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split

start = time.clock()
chinese_font = FontProperties(
    fname='/usr/share/fonts/MyFonts/YaHei.Consolas.1.11b.ttf')
plt.style.use('seaborn')
os.chdir('./data')

# load the data
input_data = pd.read_excel('baidu_index_0407.xlsx', 'BaiduIndex')
mapping_data = pd.read_excel('baidu_index_0407.xlsx', 'BaiduIndex_matching')

feature_dict = dict(zip(mapping_data['features'], mapping_data['names']))

# preprocess the data
data_clean = input_data.dropna()

# 文本数据数字化
# le = preprocessing.LabelEncoder()
# le.fit(data_clean.Season)
# data_clean.loc[:, 'Season'] = le.transform(data_clean.Season)

# 标准化数据
variables = data_clean[mapping_data['features'].tolist()]

predictors = data_clean.copy()

# for feature in mapping_data['features']:
#     predictors.loc[:, feature] = preprocessing.scale(
#         predictors[feature].astype('float64'))

# train_X = predictors.iloc[:-1, :-1]
# train_Y = data_clean.iloc[:-1, -1]
# pred_X = predictors.iloc[-1, :-1]
# 自定义标准话
X_data = predictors.iloc[:-1, :-1]
n_features = len(X_data.columns)
X = np.array(X_data.values)
y = np.array(data_clean.iloc[:-1, -1].values)
X_pred = predictors.iloc[-1, :-1]

X_standard = X.copy()
X_pred_std = X_pred.copy()
X_desc = X_data.describe()
for i_feature in range(n_features):
    mean = X_desc.iloc[1, i_feature]
    std = X_desc.iloc[2, i_feature]
    X_standard[:, i_feature] = (X_standard[:, i_feature] - mean) / std
    X_pred_std[i_feature] = (X_pred[i_feature] - mean) / std


train_X = X_standard
train_Y = data_clean.iloc[:-1, -1]
pred_X = X_pred_std

# 进行lasso回归的n_folds交叉验证,选择最好的模型
num_folds = 5
lasso_model = LassoCV(cv=num_folds).fit(train_X, train_Y)

fig1, ax1 = plt.subplots()
ax1.plot(lasso_model.alphas_, lasso_model.mse_path_, linestyle='--')

Label_Mean = "Mean of %i-flods CV" % num_folds
ax1.plot(lasso_model.alphas_, lasso_model.mse_path_.mean(
    axis=1), linewidth=2, color='purple', label=Label_Mean)
print("****MEAN RMSE: %f" % np.sqrt(lasso_model.mse_path_.mean(axis=1).min()))

Alpha_Best = lasso_model.alpha_
Label_Alpha = "Best alpha: %f" % Alpha_Best
print("****%s" % Label_Alpha)
ax1.axvline(Alpha_Best, linestyle=":", label=Label_Alpha,
            linewidth=2, color='green')

ax1.set_xlabel("Alphas", fontsize=18)
ax1.set_ylabel("EMS", fontsize=18)
ax1.semilogx()
ax1.invert_xaxis()
ax1.legend()
ax1.axis("tight")
fig1.savefig("Lasso_MSE.png", dpi=400)

# 比较系数向量
fig2, ax2 = plt.subplots()
alphas, coefs, _ = linear_model.lasso_path(train_X, train_Y, return_code=False)
alpha_index = [index for index in range(
    len(alphas)) if alphas[index] >= Alpha_Best]
alpha_selected = alphas[max(alpha_index)]

ax2.plot(alphas, coefs.T)
label_line = "The best alpha: %f" % alpha_selected
print("****The best alpha select in lasso %s" % alpha_selected)

ax2.axvline(alpha_selected, linestyle=":", label=label_line)
ax2.semilogx()
ax2.invert_xaxis()
ax2.set_xlabel("Alphas", fontsize=18)
ax2.set_ylabel("Coef values", fontsize=18)
ax2.legend()
ax2.axis('tight')
fig2.savefig("Lasso_coefs.png", dpi=400)

Importance = abs(coefs[:, list(alphas).index(alpha_selected)])
importance_pairs = sorted(zip(Importance, mapping_data['names']))
filtered_importance_pair = list(filter(lambda x: x[0] != 0, importance_pairs))
filtered_im_keys = list(map(lambda item: item[1], filtered_importance_pair))
filtered_im_values = list(map(lambda item: item[0], filtered_importance_pair))

fig3, ax3 = plt.subplots()
yticks = np.arange(len(filtered_importance_pair)) + 0.5
ax3.barh(yticks, filtered_im_values, align='center', color='purple')
ax3.set_yticks(yticks)
ax3.set_yticklabels(filtered_im_keys,
                    fontsize=12, fontproperties=chinese_font)
ax3.set_xlabel("Importance")
ax3.set_xlim(0, max(Importance))
ax3.axis("tight")
fig3.savefig("Lasso_Importance.png", dpi=400)

# 预测结果
pre_train, pre_test, tar_train, tar_test, = train_test_split(
    train_X, train_Y, test_size=.1, random_state=123)

model = linear_model.Lasso(alpha=Alpha_Best).fit(
    train_X, train_Y.values.reshape(-1, 1))

from sklearn.metrics import mean_squared_error

Train_Error = mean_squared_error(tar_train, model.predict(pre_train))
Test_Error = mean_squared_error(tar_test, model.predict(pre_test))
print('Training data MSE: %6f' % Train_Error)
print('Test data MSE: %6f' % Test_Error)

y_pred = model.predict(pred_X.values.reshape(1, -1))
print(y_pred)

Df_Importance = DataFrame(model.coef_,
                          columns=['value'],
                          index=mapping_data['names'])
Df_Importance.index.name = "features"
Df_Importance.to_excel("Total_Lasso_Feature_Importance.xlsx",
                       sheet_name="feature_importance")
