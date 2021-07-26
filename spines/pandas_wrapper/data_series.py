from spines.utils.series_op import value_type_detector
import pandas as pd
import numpy as np


class DataSeriesWrapper(pd.Series):

    """
    Determines whether each value passed to the column is an empty string
    and gets the value data type.
    params:
    col: pandas Series
    returns：
    A list containing the number of empty strings for the column
    and a set of data types for all the values of the column.
    """

    @property
    def preview(self):
        """
        返回series描述性概览
        """
        indexes=('total', 'na', 'naPercent', 'max', '75%', 'median',
                                   '25%', 'min', 'mean', 'mode', 'variation', 'std', 'skew', 'kurt', 'samples')

        na_sum = np.sum(self.isnull())
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
        null_strings, value_types, samples = value_type_detector(self)
        ds = pd.Series({
            'na': sum(na),
            'naPercent': sum(na) / len(self),
            'nullStrings': null_strings,
            'valueTypes': value_types,
            'samples': samples
        }, index=indexes)

        return DataSeriesWrapper(ds)
