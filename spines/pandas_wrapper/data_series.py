from spines import *
import pandas as pd
import numpy as np


class DataSeriesWrapper(pd.Series):

    @staticmethod
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

    @property
    def preview(self):
        """
        返回series描述性概览
        """
        indexes=('total', 'na', 'naPercent', 'max', '75%', 'median',
                                   '25%', 'min', 'mean', 'mode', 'variation', 'std', 'skew', 'kurt', 'samples')

        na_sum = self.isnull().sum()
        value_count = self.value_counts(ascending=False)
        ind_len = self.__len__()
        samples = self.sample(2).values
        if self.dtypes in (int, float, bool):
            value = {'total': ind_len,
                     'na': na_sum,
                     'naPercent': na_sum / ind_len,
                     'max': self.max(),
                     '75%': self.quantile(0.75),
                     'median': self.median(),
                     '25%': self.quantile(0.25),
                     'min': self.min(),
                     'mean': self.mean(),
                     'mode': value_count.index[0],
                     'variation': value_count.values[1:].sum() / ind_len,
                     'std': self.std(),
                     'skew': self.skew(),
                     'kurt': self.kurt(),
                     'samples': samples
                     }
        else:
            value = {'total': ind_len,
                     'na': na_sum,
                     'naPercent': na_sum / ind_len,
                     'max': np.nan,
                     '75%': np.nan,
                     'median': np.nan,
                     '25%': np.nan,
                     'min': np.nan,
                     'mean': np.nan,
                     'mode': value_count.index[0],
                     'variation': value_count.values[1:].sum() / ind_len,
                     'std': np.nan,
                     'skew': np.nan,
                     'kurt': np.nan,
                     'samples': samples
                     }
        ds = pd.Series(value, index=indexes)

        return DataSeriesWrapper(ds)

    @property
    def abnormal(self):
        """

        """
        indexes = ('na', 'naPercent', 'nullStrings', 'valueTypes', 'samples')

        na = self.isna()
        null_strings, value_types, samples = self._value_type_detector(self)
        ds = pd.Series({
            'na': sum(na),
            'naPercent': sum(na) / len(self),
            'nullStrings': null_strings,
            'valueTypes': value_types,
            'samples': samples
        }, index=indexes)

        return DataSeriesWrapper(ds)
