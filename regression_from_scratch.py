## Linear Regression using only numpy
import numpy as np
from read_data import get_data

from sklearn.metrics import mean_squared_error, r2_score

def fit_line():
    x_train, y_train, x_test, y_test = get_data()

    n = len(x_train)  # Number of training samples
    alpha = 0.0002

    a_0 = 1
    a_1 = 1

    epochs = 0
    while epochs < 1000:
        # Shuffle the training data
        indices = np.arange(n)
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        # Stochastic gradient descent
        for i in range(n):
            x_i = x_train[i]
            y_i = y_train[i]

            y_pred = a_0 + a_1 * x_i
            error = y_pred - y_i

            # Update coefficients
            a_0 -= alpha * error
            a_1 -= alpha * error * x_i
        
        # Calculate mean squared error over the entire dataset
        y_pred_all = a_0 + a_1 * x_train
        mean_sq_er = np.mean((y_pred_all - y_train) ** 2)

        epochs += 1

        if epochs % 10 == 0:
            print(f"Epoch: {epochs}, Mean Squared Error: {mean_sq_er:.4f}")

    # With stochastic gradient descent, we have an r2 score of 98.567%
    y_pred = a_0 + a_1 * x_test
    print(r2_score(y_test,y_pred))

    return a_0, a_1

if __name__ == '__main__':
    fit_line()