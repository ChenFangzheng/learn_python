import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model


def get_data(file_name):

    data = pd.read_csv(file_name)
    X_parameter = []
    Y_parameter = []

    for single_square_feet, single_price_value in zip(data['square_feet'], data['price']):
        X_parameter.append(single_square_feet)
        Y_parameter.append(single_price_value)

    return X_parameter, Y_parameter


def linear_model_main(X_parameter, Y_parameter, predict_value):
    regr = linear_model.LinearRegression()
    regr.fit(X_parameter, Y_parameter)
    predict_outcome = regr.predict(predict_value)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome

    return predictions


def show_linear_line(X_parameter, Y_parameter):
    regr = linear_model.LinearRegression()
    regr.fit(X_parameter, Y_parameter)
    plt.scatter(X_parameter, Y_parameter, color='blue')
    plt.plot(X_parameter, regr.predict(X_parameter), color='red', linewidth=4)
    # plt.xticks(())
    # plt.yticks(())
    plt.show()


X, Y = get_data('input_data.csv')
predictvalue = 700
result = linear_model_main(np.array(X).reshape(-1, 1), Y, predictvalue)
print('intercept ', result['intercept'])
print('coefficient ', result['coefficient'])
print('predicted_value ', result['predicted_value'])

show_linear_line(np.array(X).reshape(-1, 1), Y)
