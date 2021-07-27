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
        weights: Weighted moving average, the greater the list index, the later the date.
        return:
        pd.Series, the predict values.
        """
        assert isinstance(series, pd.Series), "series must be pandas Series."
        assert isinstance(period, int), "period must be integer."
        assert isinstance(weights, list) or weights is None, "weights must be None or List."
        if isinstance(weights, list) is True:
            assert len(weights) == period, "weights length must be equals to period."

        if weights is not None:
            res = [np.nan for i in range(period)]
            _ = []
            for i in list(series.rolling(period)):
                if len(i) == period:
                    _.append(i.values)
                    
            assert len(_) > 0, "List length must be greater than 0."
            _ = np.round(np.sum(np.stack(_, axis=0) * weights, axis=1), 2)
            res.extend(_)

            return res
        else:
            res = series.rolling(period+1).mean().to_list()
            res.append(np.mean([res[-1], res[-2]]))
            return res

