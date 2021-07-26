import pandas as pd
import numpy as np
from spines.utils.series_op import value_type_detector
import seaborn as sns
import matplotlib.pyplot as plt


class DataFrameWrapper(pd.DataFrame):
    """
    Define a class, inherit Pandas.DataFrame,
    facilitate viewing data and other EDA customized data viewing requirements.
    """

    @staticmethod
    def _plot_na(pv: pd.DataFrame, percentage: bool = True):
        sns.set()
        if percentage:
            target_col = 'naPercent'
        else:
            target_col = 'na'

        _ = pv.query('na > 0')[target_col].sort_values(ascending=True)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        _.plot(kind='barh', figsize=(12, 10), title=f'Plot {target_col}', ax=ax1)
        if percentage:
            for a, b in enumerate(_.values):
                ax1.text(b, a, str(round(b * 100, 1)) + '%')
        else:
            for a, b in enumerate(_.values):
                ax1.text(b, a, str(round(b, 2)))

        _.plot(kind='barh', figsize=(12, 10), title=f'Plot dtypes', ax=ax2)
        ax2.yaxis.tick_right()
        for a, b in enumerate(_.values):
            ax2.text(b, a, pv['valueTypes'][a])
        plt.show()

    @property
    def preview(self):
        """
        For data previews, to check various data properties of the dataset.
        returns:
        total: number of elements
        na: null values
        naPercent: null value accounts for this ratio
        dtype: datatype
        75%: 75% quantile
        25%: 25% quantile
        variation: variation ratio
        std: standard deviation
        skew: Skewness
        kurt: Kurtosis
        samples: Random returns two values
        """

        col = self.columns
        ind_len = self.shape[0]

        df = pd.DataFrame(columns=('total', 'na', 'naPercent', 'dtype', 'max', '75%', 'median',
                                   '25%', 'min', 'mean', 'mode', 'variation', 'std', 'skew', 'kurt', 'samples'))

        pointer = 0
        for i in col:
            samples = ' || '.join([str(self[i][j]) for j in np.random.randint(ind_len, size=2)])
            na_sum = self[i].isnull().sum()
            value_count = self[i].value_counts(ascending=False)
            if self[i].dtypes in (int, float, bool):
                value = {'total': ind_len,
                         'na': na_sum,
                         'naPercent': na_sum / ind_len,
                         'dtype': self[i].dtype,
                         'max': self[i].max(),
                         '75%': self[i].quantile(0.75),
                         'median': self[i].median(),
                         '25%': self[i].quantile(0.25),
                         'min': self[i].min(),
                         'mean': self[i].mean(),
                         'mode': value_count.index[0],
                         'variation': value_count.values[1:].sum() / ind_len,
                         'std': self[i].std(),
                         'skew': self[i].skew(),
                         'kurt': self[i].kurt(),
                         'samples': samples
                         }
            else:
                value = {'total': ind_len,
                         'na': na_sum,
                         'naPercent': na_sum / ind_len,
                         'dtype': self[i].dtype,
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
            df.loc[i] = pd.Series(value, name=i)
            pointer += 1

        if df.shape[0] < df.shape[1]:
            return DataFrameWrapper(df.T)
        else:
            return DataFrameWrapper(df)

    @property
    def abnormal(self):
        """
        View the outliers of the data set.
        returns： pd.DataFrame
        na: Number of null values of this column
        nullStrings: Number of empty strings
        valueTypes：Datatype set for all values in this column
        """
        df = pd.DataFrame(columns=('na', 'naPercent', 'nullStrings', 'valueTypes', 'samples'))

        for i in self.columns:
            na = self[i].isna()
            null_strings, value_types, samples = value_type_detector(self[i])
            df.loc[i] = pd.Series({
                'na': sum(na),
                'naPercent': sum(na) / len(self[i]),
                'nullStrings': null_strings,
                'valueTypes': value_types,
                'samples': samples
            }, name=i)

        return DataFrameWrapper(df)

    def abnormal_plot(self, plot_na_percent=True):
        """
        plot outliers distribute.
        plot_na_percent: whether plot the percent of nan value .
        return: None
        """
        df = self.abnormal
        if plot_na_percent:
            self._plot_na(df, percentage=True)
        else:
            self._plot_na(df, percentage=False)


def compare_shape(*df):
    """Compare Pandas DataFrames shapes."""
    assert all([isinstance(i, pd.DataFrame) for i in df]) is True, \
        "The parameters must be pandas DataFrames."

    shapes = [d.shape for d in df]
    shape_row = set([i[0] for i in shapes])
    shape_col = set([i[1] for i in shapes])

    print(f'The rows of DataFrames are equal: {"yes" if len(shape_row) == 1 else "no"}')
    print(f'The cols of DataFrames are equal: {"yes" if len(shape_col) == 1 else "no"}')
    return shapes


def num_of_na(df):
    """
    df: pandas DataFrame
    return:
    the number of null values in Pandas DataFrame.
    """
    null_details = np.sum(df.isna())
    null_sum = np.sum(null_details)
    print(f"Total null value number is : {null_sum}")
    if null_sum > 0:
        res = {}
        for k, v in null_details.iteritems():
            if v > 0:
                res[k] = v
        return res
    return
