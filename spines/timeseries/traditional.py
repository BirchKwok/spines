import pandas as pd
import numpy as np

class Traditional:
    """
    Traditional methods for time series.
    """
    @staticmethod
    def smooth(series: pd.Series, a):
        """
        series: pd.Series
        a: last day smooth values ratio.
        """
        init_value = series[0] * 1.05
        res = [init_value]
        for i in range(1, len(series)):
            s_i = a * series[i-1] + (1 - a)*res[i-1]
            res.append(s_i)
        return res

    @staticmethod
    def moved_avg(series: pd.Series, period=2, weights: list = None):
        """
        moving average method.
        series: pd.Series
        period: predict period
        weights:
        return:
        pd.Series
        """
        assert isinstance(period, int), "period must be integer."
        if isinstance(weights, list) is True:
            assert len(weights) == period, "weights length must be equals to period."

        if weights is not None:
            res = [np.nan for i in range(period - 1)]
            _ = []
            for i in list(series.rolling(period)):
                if len(i) == period:
                    _.append(i.values)

            for j in range(len(_)):
                t = []
                for i, c in zip(weights, _[j]):
                    t.append(c * i)
                res.append(np.round(np.mean(t), 3))

            return res
        else:
            return series.rolling(period).mean().to_list()

