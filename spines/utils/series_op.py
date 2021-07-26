import pandas as pd


def value_type_detector(col:pd.Series):
    """
    Determines whether each value passed to the column is an empty string
    and gets the value data type.
    params:
    col: pandas Series
    returns：
    A list containing the number of empty strings for the column
    and a set of data types for all the values of the column.
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



