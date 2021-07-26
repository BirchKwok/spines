

class 
def smooth(series, a):
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