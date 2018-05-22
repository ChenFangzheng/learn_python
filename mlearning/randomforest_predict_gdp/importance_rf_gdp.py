from collections import defaultdict
import numpy as np
import random
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine

# matplotlib基础设置
plt.style.use('seaborn')
plt.rc('figure', figsize=(10, 7))
rcParams.update({'figure.autolayout': True})

# 获取数据
ENGINE = create_engine(
    'mysql+mysqlconnector://root:Pass@word@127.0.0.1/economic_forecast')

TABLE_NAME = 't_nanning_baiduindex_v3'
DATAFRAME = pd.read_sql_table(TABLE_NAME, ENGINE)
FEATURES = list(DATAFRAME.keys())[3:-4]
predictors = DATAFRAME.copy()
predictors = predictors.iloc[:, 3:-3]

# 对训练数据的集的自变量做标准化
X_data = predictors.iloc[:-1, :-1]
X_pred = predictors.iloc[-1, :-1]
y_data = DATAFRAME.iloc[:-1, -4]

#　构建随机森林，预测数据
model_rf = RandomForestRegressor(n_estimators=5000, oob_score=True,
                                 n_jobs=-1, random_state=500, max_features="auto")
model_rf.fit(X_data, y_data)

y_pred_all = model_rf.predict(X_data)
print("The RMSE for all samples: %.4f" %
      np.sqrt(mean_squared_error(y_data, y_pred_all)))
print("The prediction is: %.4f" %
      model_rf.predict(X_pred.values.reshape(1, -1)))

# 输出预测结果
cmp_arr = np.zeros(2 * len(y_data)).reshape(2, len(y_data))
cmp_arr[0] = y_data
cmp_arr[1] = y_pred_all
pred_df = DataFrame(cmp_arr.T,
                    columns=['True Value', 'Predict Vlaue'], index=DATAFRAME.iloc[:-1, 2])
pred_df["Error"] = y_pred_all - np.array(y_data)
pred_df.to_excel('Total_pred_result.xlsx',
                 sheet_name='pred_result')

# 特征重要性列表 Mean decrease impurity
DataFrame(model_rf.feature_importances_,
          columns=['Importance'], index=FEATURES)\
    .to_excel('Total_feature_importance.xlsx',
              sheet_name='RF_feature_im_')


# 特征重要性列表 Mean decrease accuracy
# scores = defaultdict(list)
# rs = ShuffleSplit(n_splits=len(X_data), test_size=0.3, random_state=100)

# for train_idx, test_idx in rs.split(X_data):
#     X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
#     Y_train, Y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]
#     r = model_rf.fit(X_train, Y_train)
#     acc = r2_score(Y_test, r.predict(X_test))
#     for i in range(X_data.shape[1]):
#         X_t = X_test.copy()
#         # np.random.shuffle(X_t.iloc[:, i])
#         X_t.iloc[:, i] = 1
#         shuff_acc = r2_score(Y_test, r.predict(X_t))
#         scores[FEATURES[i]].append((acc - shuff_acc) / acc)

# importance2 = sorted([(feat, abs((round(np.mean(score), 4))))
#                       for feat, score in scores.items()], reverse=True)
# df_im2=DataFrame(list(importance2),
#                    columns=['feature name', 'weight'])
# df_im2.to_excel('Feature_importance2.xlsx', sheet_name='importance2')
