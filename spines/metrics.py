from . import *


class RegModelEvaluate:
    """regression model evaluate functions."""

    def r_square(self, y, y_hat):
        return 1 - (np.sum(np.power(y - y_hat, 2) )/ np.sum(np.power(y - np.mean(y), 2)))

    def rmse(self, y, y_hat):
        return np.sqrt( 1 /len(y) * np.sum(y - y_hat) ** 2)

    def mae(self, y, y_hat):
        return 1 / len(y) * np.sum(np.abs( y -y_hat))

    def mape(self, y, y_hat):
        return 1/ len(y) * np.sum(np.abs((y - y_hat) / y))