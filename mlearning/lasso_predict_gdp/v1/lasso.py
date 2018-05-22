'''
深圳经济数据预测：GDP
'''
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LassoCV  # 导入LassoCV算法包
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

rcParams.update({'figure.autolayout': True})

# 获取数据
ENGINE = create_engine(
    'mysql+mysqlconnector://root:Pass@word@127.0.0.1/economic_forecast')

TABLE_NAME = 't_shenzhen_baiduindex_v1.0'
DATAFRAME = pd.read_sql_table(TABLE_NAME, ENGINE)
FEATURES = list(DATAFRAME.keys())[3:-4]
predictors = DATAFRAME.copy()
predictors = predictors.iloc[:, 3:-3]

# 对训练数据的集的自变量做标准化
X_data = predictors.iloc[:-1, :-1]
X_pred = predictors.iloc[-1, :-1]
y_data = DATAFRAME.iloc[:-1, -4]
X_desc = X_data.describe()

for feature_index in range(len(FEATURES)):
    mean = X_desc.iloc[1, feature_index]
    std = X_desc.iloc[2, feature_index]
    X_data.iloc[:, feature_index] = (
        (X_data.iloc[:, feature_index] - mean) / std)
    X_pred.iloc[feature_index] = (X_pred.iloc[feature_index] - mean) / std
scaler = preprocessing.StandardScaler()
scaler.fit(predictors)
scaled_data = scaler.transform(predictors)


# for feature in FEATURES:
#     predictors.loc[:, feature] = preprocessing.scale(
#         predictors[feature].astype('float64'))

# 训练数据集
# X_data = scaled_data[:-1, :]
# y_data = DATAFRAME.iloc[:-1, -3]
# 需预测数据的自变量
# X_pred = scaled_data[-1, :]

# 进行lasso回归的n_folds交叉验证
NUM_FOLDS = 5
FIG1, AX1 = plt.subplots()
lasso_model = LassoCV(cv=NUM_FOLDS).fit(X_data, y_data)
ALPHA_BEST = lasso_model.alpha_
MEAN_RMSE = lasso_model.mse_path_.mean(axis=1)
print("%s The best alpha select by LassoCV is %f" % ('*' * 5, ALPHA_BEST))
print("Mean RMSE: %f" % np.sqrt(MEAN_RMSE).min())
print(lasso_model.predict(X_pred.reshape(1, -1)))

LABEL_MEAN = "Mean of %i-folds CV" % NUM_FOLDS
LABEL_ALPHA = "Best alpha: %f" % ALPHA_BEST

AX1.plot(lasso_model.alphas_, lasso_model.mse_path_, linestyle="--")
AX1.plot(lasso_model.alphas_, MEAN_RMSE, linewidth=2,
         color="purple", label=LABEL_MEAN)
AX1.axvline(ALPHA_BEST, linestyle=":", label=LABEL_ALPHA,
            linewidth=2, color="green")
AX1.set_xlabel("Alphas", fontsize=18)
AX1.set_ylabel("EMS", fontsize=18)
AX1.semilogx()
AX1.invert_xaxis()
AX1.legend()
AX1.axis("tight")
FIG1.savefig("Lasso_MSE.png", dpi=400)

# 选择模型（系数向量）
FIG2, AX2 = plt.subplots()
ALPHAS, COEFS, _ = linear_model.lasso_path(X_data, y_data, return_n_iter=False)
ALPHA_INDEX = [index for index in range(
    len(ALPHAS)) if ALPHAS[index] >= ALPHA_BEST]
ALPHA_SELECTED = ALPHAS[max(ALPHA_INDEX)]
print("%sThe best alpha select by lasso_path: %f" % ('*' * 5, ALPHA_SELECTED))

LABELE_LINE = "The best alpha: %f" % ALPHA_SELECTED
AX2.plot(ALPHAS, COEFS.T)
AX2.axvline(ALPHA_SELECTED, linestyle=":", label=LABELE_LINE)
AX2.semilogx()
AX2.invert_xaxis()
AX2.set_xlabel("Alphas", fontsize=18)
AX2.set_ylabel("coef values", fontsize=18)
AX2.legend()
AX2.axis("tight")
FIG2.savefig("Lasso_coefs.png", dpi=400)

IMPORTANCE = abs(COEFS[:, max(ALPHA_INDEX)])
IMPORTANCE_PAIRS = list(zip(IMPORTANCE, FEATURES))
IMPORTANCE_FILTERED = [im_pair for im_pair in IMPORTANCE_PAIRS
                       if im_pair[0] > 0]
IMPORTANCE_SORTED = list(sorted(IMPORTANCE_FILTERED))
IMPORTANCE_KEYS = list(map(lambda item: item[1], IMPORTANCE_SORTED))
IMPORTANCE_VALUES = list(map(lambda item: item[0], IMPORTANCE_SORTED))

FIG3, AX3 = plt.subplots()
Y_TICKS = np.arange(len(IMPORTANCE_FILTERED)) + 0.5
AX3.barh(Y_TICKS, IMPORTANCE_VALUES, align='center', color='purple')
AX3.set_yticks(Y_TICKS)
AX3.set_yticklabels(IMPORTANCE_KEYS, fontsize=12)
AX3.set_xlabel("Importance")
AX3.set_xlim(0, max(IMPORTANCE_VALUES))
AX3.axis('tight')
FIG3.savefig('Lasso_Importance.png', dpi=400)

# 预测结果
REG_LASSO = linear_model.Lasso(alpha=ALPHA_BEST).fit(X_data,
                                                     y_data.values.reshape(-1, 1))

PRE_TRAIN, PRE_TEST, TAR_TRAIN, TAR_TEST, = train_test_split(
    X_data, y_data, test_size=.1, random_state=123)

from sklearn.metrics import mean_squared_error

TRAIN_ERROR = mean_squared_error(TAR_TRAIN, REG_LASSO.predict(PRE_TRAIN))
TEST_ERROR = mean_squared_error(TAR_TEST, REG_LASSO.predict(PRE_TEST))
print('Training data MSE: %6f' % TRAIN_ERROR)
print('Test data MSE: %6f' % TEST_ERROR)

Y_PRED = REG_LASSO.predict(X_pred.reshape(1, -1))

print("预测结果： %f" % Y_PRED[0])

cmp_arr = np.zeros(2 * len(y_data)).reshape(2, len(y_data))
cmp_arr[0] = y_data
cmp_arr[1] = REG_LASSO.predict(X_data)
PRED_DF = DataFrame(cmp_arr.T, columns=["True Value", "Predicted Value"],
                    index=DATAFRAME.iloc[:-1, 2])

# PRED_DF.plot()
# xticks = range(len(y_data))
# plt.xticks(xticks, PRED_DF.index, fontsize=10, rotation=30)
# plt.xlabel("Season", fontsize=15)
# plt.ylabel("GDP Speed", fontsize=15)

PRED_DF.to_excel("Total_Lasso_Predict_Result.xlsx",
                 sheet_name="Lasso_Predict_Result")

DF_IMPORTANCE = DataFrame(REG_LASSO.coef_, columns=['Value'], index=FEATURES)
DF_IMPORTANCE.to_excel("Total_Lasso_Feature_Importance.xlsx",
                       sheet_name="feature_importance")
