import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import sklearn.linear_model 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from linear_model import LinearModel


class LinearRegression(LinearModel):
    """Linear regression with Gradient Descent as the solver.

    Example usage(clf aka classifier): 
        > clf = LinearRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=100, batch_grad=True):
        super().__init__(step_size, max_iter)
        self.batch_grad = batch_grad
        self.theta = None 

    def batch_gradient_descent(self, x, y):
        h_x = np.dot(x, self.theta)    # hypotheses h 
        grad = np.dot((h_x - y), x) 
        return grad
    
    def stochastic_gradient_descent(self, x_i, y_i):
        h_x = np.dot(x_i, self.theta)
        grad = (h_x - y_i) * x_i
        return grad

    def fit(self, x, y):
        """Run Gradient Descent to minimize J(theta) for Linear regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # let x_0 = 1 (intercept term 截距)
        x = np.insert(x, 0, 1, axis=1)
        n_samples, n_features = x.shape
        
        # init parameters
        self.theta = np.zeros(n_features)
        self.training_error = []
        self.hypothese = []

        # Least-Mean-Square update rule
        for i in range(self.max_iter):
            if self.batch_grad:
                grad = self.batch_gradient_descent(x, y)
                
            else:
                # stochastic gradient descent
                i = i % n_samples
                grad = self.stochastic_gradient_descent(x[i], y[i])

            # update parameters
            self.theta -= self.step_size*grad

            # record training loss in each iteration
            h_x = np.dot(x, self.theta)
            mse = np.mean(0.5 * (h_x-y)**2)
            self.training_error.append(mse)
            self.hypothese.append(h_x)

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        x = np.insert(x, 0, 1, axis=1)
        y_pred = np.dot(x, self.theta)
        return y_pred


if __name__ == "__main__":
    X, y = make_regression(n_samples=200, n_features=1, noise=10, bias=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # self implemented
    clf = LinearRegression(step_size=0.01, max_iter=200, batch_grad=True)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    y_pred_line = clf.predict(X)
    print('Theta = {}'.format(clf.theta))

    # compare with sklearn library
    reg = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
    print('Intercept = {1}, Theta_0 = {0}'.format(reg.coef_, reg.intercept_))

    # Training error plot
    n = len(clf.training_error)
    training, = plt.plot(range(n), clf.training_error, label="Training Error")
    plt.legend(handles=[training])
    plt.title("Error Plot")
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Iterations')
    plt.show()
    
    # plot scatter X,y and fitted line
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()

    # animation gif
    intermediate = clf.hypothese
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("viridis")
    
    def init():
        ax.plot(X_train, 0*X_train)

    def animation_func(i):
        ax.clear()
        ax.plot(X_train, intermediate[i])
        m1 = ax.scatter(X_train, y_train, color=cmap(0.9), s=10)
        m2 = ax.scatter(X_test, y_test, color=cmap(0.5), s=10)
        ax.set_xlim([0,2])
        ax.set_ylim([0,200])

    ani = FuncAnimation(fig, animation_func, init_func=init, frames=(len(intermediate)),interval = 20)
    ani.save('./animate.gif',fps=1)