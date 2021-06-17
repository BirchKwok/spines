from . import *


def _value_type_detector(col):
    """
    判断传入列的每个值是否为空字符串，并获取该值数据类型。
    params:
    col: pandas Series
    returns：
    一个列表，包含该列的空字符串数目、以及一个该列所有值的数据类型的set集合。
    """
    count = 0
    types = list()
    random_results = dict()
    for j in col:
        if isinstance(j, str) and j.strip() == "":
            count += 1
        type_j = type(j)
        if type_j not in types:
            types.append(type_j)
        if type_j.__name__ not in random_results.keys():
            random_results[type_j.__name__] = 0

    for k in random_results.keys():
        for t in types:
            random_results[k] = pd.Series([i for i in col if isinstance(i, t)]).sample(2).values

    return count, [i.__name__ for i in types], random_results
