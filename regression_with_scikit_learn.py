from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from read_data import get_data

def get_y_pred():

    x_train, y_train, x_test, y_test = get_data()

    regr = linear_model.LinearRegression()

    regr.fit(x_train, y_train)

    y_pred = regr.predict(x_test)

    print(r2_score(y_test,y_pred))

    return y_pred

if __name__ == '__main__':
    get_y_pred()