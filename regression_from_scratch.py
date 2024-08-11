## Linear Regression using only numpy
import numpy as np
from read_data import get_data

def fit_line():
    x_train, y_train, _, _ = get_data()

    n = len(x_train)  # Number of training samples
    alpha = 0.0002

    a_0 = np.random.rand()
    a_1 = np.random.rand()

    epochs = 0
    while epochs < 5000:
        y = a_0 + a_1 * x_train # a linear regression analog of forward propagation
        error = y - y_train
        mean_sq_er = np.mean(error ** 2)  # Mean squared error
        
        if epochs % 100 == 0:
            print(f"Epoch: {epochs}, Mean Squared Error: {mean_sq_er}")
        
        # Update coefficients - a linear regression analog of back propagation to change weights
        a_0 -= alpha * 2 * np.mean(error)
        a_1 -= alpha * 2 * np.mean(error * x_train)
        
        epochs += 1

    return a_0, a_1

if __name__ == '__main__':
    fit_line()