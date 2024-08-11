import matplotlib.pyplot as plt
from regression_from_scratch import fit_line

from read_data import get_data
from regression_with_scikit_learn import get_y_pred

def plot_regression_analysis():

    _, _, x_test, y_test = get_data()

    a_0, a_1 = fit_line()

    y_plot = []
    for i in range(100):
        y_plot.append(a_0 + a_1 * i)
    
    plt.figure(figsize=(10,10))
    plt.scatter(x_test,y_test,color='black',label='data')
    plt.plot(range(len(y_plot)),y_plot,color='blue', label = 'predict')

    plt.xticks(())
    plt.yticks(())
    plt.legend()
    
    plt.show()

def plot_from_sklearn():
    
    _, _, x_test, y_test = get_data()
    y_pred = get_y_pred()
    plt.scatter(x_test, y_test, color = 'black', label = 'data')
    plt.plot(x_test, y_pred, color = 'blue', label = 'predict')

    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_regression_analysis()