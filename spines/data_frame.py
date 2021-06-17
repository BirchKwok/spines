from . import *


class DataFrameWrapper(pd.DataFrame):
    """
    定义一个类，继承pandas.DataFrame，方便查看数据和其他EDA定制化数据查看需求。
    """
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
        用于数据预览，查看数据集的各种数据属性。
        returns:
        total: 数量,
        na: 空值,
        naPercent: 空值所占该列比例,
        dtype: 该列的数据类型(dtype),
        max: 该列最大值,
        75%: 75%分位数,
        median: 中位数,
        25%: 25%分位数,
        min:该列最小值,
        mean: 均值,
        mode: 众数,
        variation: 异众比率,
        std: 标准差,
        skew: 偏度系数,
        kurt: 峰度系数,
        samples: 随机返回该列两个值
        """
        # ind = self.index
        col = self.columns
        ind_len = self.shape[0]
        # col_len = len(col)
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
        查看数据集的异常值情况。
        returns：
        na: 该列的空值数目,
        nullStrings: 该列空字符串数目,
        valueTypes：该列所有数值的数据类型set集合
        """
        df = pd.DataFrame(columns=('na', 'naPercent', 'nullStrings', 'valueTypes', 'samples'))

        for i in self.columns:
            na = self[i].isna()
            null_strings, value_types, samples = self._value_type_detector(self[i])
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
        plot abnormal value distribute .

        :param plot_na_percent: whether plot the percent of nan value .
        :return: None
        """
        df = self.abnormal
        if plot_na_percent:
            self._plot_na(df, percentage=True)
        else:
            self._plot_na(df, percentage=False)


def compare_shape(old_df_name, new_df_name):
    """compare old and new dataset shape"""
    if not (isinstance(old_df_name, pd.DataFrame) and isinstance(new_df_name, pd.DataFrame)):
        raise TypeError("the parameters must be pandas data frames.")
    a = old_df_name.shape
    b = new_df_name.shape
    print(f'The rows of two data frames are equal: {a[0] == b[0]}')
    return a, b


def num_of_na(df):
    """return data frame null value number."""
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


















