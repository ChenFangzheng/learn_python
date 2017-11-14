import numpy as np
import random
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine

# matplotlib基础设置
plt.style.use('seaborn')
plt.rc('figure', figsize=(10, 7))
rcParams.update({'figure.autolayout': True})

# 获取数据
ENGINE = create_engine(
    'mysql+mysqlconnector://root:Pass@word@127.0.0.1/economic_forecast')

TABLE_NAME = 't_shenzhen_baiduindex_v1.0'
DATAFRAME = pd.read_sql_table(TABLE_NAME, ENGINE)
FEATURES = list(DATAFRAME.keys())[3:-3]
predictors = DATAFRAME.copy()
predictors = predictors.iloc[:, 3:-2]

# 对训练数据的集的自变量做标准化
X_data = predictors.iloc[:-1, :-1]
X_pred = predictors.iloc[-1, :-1]
y_data = DATAFRAME.iloc[:-1, -3]

#　构建随机森林，预测数据
model_rf = RandomForestRegressor(n_estimators=500, oob_score=True,
                                 n_jobs=-1, random_state=50, max_features="auto")
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

# 特征重要性列表
DataFrame(model_rf.feature_importances_,
          columns=['Importance'], index=FEATURES)\
    .to_excel('Total_feature_importance.xlsx',
              sheet_name='RF_feature_im_')
