# Linear Regression

Comparing various methods of linear regression, including its use to fit nonlinear data. This is great at getting me to understand the math behind so many ML algorithms!

Additionally, I have implemented those methods both from scratch (i.e. only numpy) and using other libraries such as scikit-learn.

The data comes from two csv files, which are processed and fitted. Finally, the resulting line from the regression is plotted using matplotlib.

## Algorithm Explanation

The initial implementation of the regression from scratch used batch gradient descent, which was horrible. During each epoch, it took all 700 data points, and changed the weights according to the average error of all data points. This resulted in a badly fitted line.

Now, I have implemented it with stochastic gradient descent. The algorithm goes through all data points iteratively, and updates its weights according to each one. 
Finally, at the end of each epoch out of 1000 epochs, it computes the mean squared error.

## File stucture

The file `read_data.py` collects the data from the csv files `train.csv` and `test.csv` and forms numpy arrays out of their respective x and y columns.

The file `regression_from_scratch.py` fits the line using least squares linear regression by stochastic gradient descent.

The file `regression_from_scikit_learn.py` fits the line using the scikit-learn library, just to allow me to see what the perfect implementation looks like

The file `plot_result.py` plots the predicted line from a fitting algorithm used.

## Credits

The data set comes from a Kaggle dataset found [here](https://www.kaggle.com/datasets/andonians/random-linear-regression/data). This was a rather faulty set, since one y-value was missing, causing completely wacky values to appear. To fix this, I manually entered a new data point.

The data collection, plotting, and regression from scratch files were inspired by [this](https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a) tutorial. Again, however, the algorithm was very faulty, and required numerous fix ups, until it got to this one with SGD which finally works. 