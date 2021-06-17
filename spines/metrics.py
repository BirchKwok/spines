from . import *


class RegModelEvaluate:
    """regression model evaluate functions."""
    @staticmethod
    def r_square(y, y_hat):
        return 1 - (np.sum(np.power(y - y_hat, 2) )/ np.sum(np.power(y - np.mean(y), 2)))

    @staticmethod
    def rmse(y, y_hat):
        return np.sqrt( 1 /len(y) * np.sum(y - y_hat) ** 2)

    @staticmethod
    def mae(y, y_hat):
        return 1 / len(y) * np.sum(np.abs( y -y_hat))

    @staticmethod
    def mape(y, y_hat):
        return 1/ len(y) * np.sum(np.abs((y - y_hat) / y))
