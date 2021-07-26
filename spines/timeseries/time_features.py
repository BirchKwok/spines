import pandas as pd
import chinese_calendar as calendar
import datetime


class FeatureBuilder:
    """Build Features. """
    @staticmethod
    def add_timestamp(df, ts_col) -> pd.DataFrame:
        """
        get the timestamp and chinese festival.

        ts_col:
        time series column,and only take one column.

        return: pandas.DataFrame

        """
        if isinstance(df[ts_col].dtypes, object):
            df[ts_col] = pd.to_datetime(df[ts_col])
        df['day'] = df[ts_col].dt.day
        df['month'] = df[ts_col].dt.month
        df['year'] = df[ts_col].dt.year
        df['week'] = df[ts_col].dt.week
        df['weekday'] = df[ts_col].dt.weekday
        df['festival_name'] = df[ts_col].apply(lambda s:
                                               calendar.get_holiday_detail(s)[1]
                                               if calendar.get_holiday_detail(s)[0] is True else None)
        return df

    @staticmethod
    def get_legal_hday(start=datetime.date(2020, 1, 1), end=datetime.date(2021, 12, 12), get_dict='df',
                       include_weekends=False):
        """get chinese legal holiday."""

        legal_hday = pd.DataFrame()
        hday = calendar.get_holidays(
            start=start, end=end, include_weekends=include_weekends)
        for j in [[i, calendar.get_holiday_detail(i)[1]] for i in hday]:
            legal_hday = legal_hday.append(pd.DataFrame(
                {'fest_name': j[1], 'date': j[0]}, index=[0]), ignore_index=True)
        if get_dict == 'df':
            return legal_hday
        else:
            legal_hday_dict = dict()
            for i in FeatureBuilder().get_legal_hday(start=start, end=end, include_weekends=include_weekends)['fest_name'].unique():
                legal_hday_dict[i] = set()
            for index, row in FeatureBuilder().get_legal_hday(start=start, end=end, include_weekends=include_weekends).iterrows():
                for k in legal_hday_dict.keys():
                    if row['fest_name'] == k:
                        legal_hday_dict[k].add(row['date'])
            return legal_hday_dict

    @staticmethod
    def extract_is_feature(df, ts_col) -> pd.DataFrame:
        """extract the is feature."""
        total_dataset = df.copy(deep=True)

        # 是否为周末
        total_dataset['is_weekend'] = 0
        total_dataset.loc[total_dataset['weekday'].isin(
            {5, 6}), 'is_weekend'] = 1

        # 是否为假期
        total_dataset['is_holiday'] = 0
        total_dataset['is_holiday'] = total_dataset[ts_col].apply(
            lambda s: 1 if calendar.is_holiday(s) else 0)

        # 是否为节假日第一天
        last_day_flag = 0
        total_dataset['is_first_of_holiday'] = 0
        for index, row in total_dataset.iterrows():
            if last_day_flag == 0 and row['is_holiday'] == 1:
                total_dataset.loc[index, 'is_first_of_holiday'] = 1
            last_day_flag = row['is_holiday']

        # 是否为节假日最后一天
        total_dataset['is_lastday_of_holiday'] = 0
        for index, row in total_dataset.iterrows():
            # 如果是最后一行，并且前一天是节假日
            if index == len(total_dataset) - 1:

                # 如果下一个工作日刚好是明天，并且今天是节假日
                if total_dataset.loc[index, 'is_holiday'] == 1 and (
                        (row[ts_col].to_pydatetime() + datetime.timedelta(days=1)) ==
                        calendar.find_workday(date=row[ts_col].to_pydatetime())):
                    total_dataset.loc[index, 'is_lastday_of_holiday'] = 1

            elif row['is_holiday'] == 1 and total_dataset.loc[index + 1, 'is_holiday'] == 0:
                total_dataset.loc[index, 'is_lastday_of_holiday'] = 1

        # 是否为节假日后的上班第一天
        total_dataset['is_firstday_of_work'] = 0
        last_day_flag = 0
        for index, row in total_dataset.iterrows():
            if last_day_flag == 1 and row['is_holiday'] == 0:
                total_dataset.loc[index, 'is_firstday_of_work'] = 1
            last_day_flag = row['is_lastday_of_holiday']

        # 是否不用上班
        total_dataset['is_work'] = 0
        total_dataset['is_work'] = total_dataset[ts_col].apply(
            lambda s: 1 if calendar.is_workday(s) else 0)

        # 是否明天要上班
        total_dataset['is_gonna_work_tomorrow'] = 0
        for index, row in total_dataset.iterrows():
            if calendar.is_workday(
                    row[ts_col].to_pydatetime() + datetime.timedelta(days=1)):
                total_dataset.loc[index, 'is_gonna_work_tomorrow'] = 1

        # 昨天上班了吗
        total_dataset['is_work_yesterday'] = 0
        for index, row in total_dataset.iterrows():
            if calendar.is_workday(
                    row[ts_col].to_pydatetime() - datetime.timedelta(days=1)):
                total_dataset.loc[index, 'is_work_yesterday'] = 1

        # 是否是放假前一天
        total_dataset['is_lastday_of_work'] = 0
        for index, row in total_dataset.iterrows():
            if row['is_holiday'] == 0 and calendar.is_holiday(
                    row[ts_col].to_pydatetime() + datetime.timedelta(days=1)) == True:
                total_dataset.loc[index, 'is_lastday_of_work'] = 1

        # 是否为月初第一天
        total_dataset['is_firstday_of_month'] = 0
        total_dataset.loc[total_dataset['day'] == 1, 'is_firstday_of_month'] = 1

        # 是否为月初第二天
        total_dataset['is_secday_of_month'] = 0
        total_dataset.loc[total_dataset['day'] == 2, 'is_secday_of_month'] = 1

        # 是否为月初
        total_dataset['is_premonth'] = 0
        total_dataset.loc[total_dataset['day'] <= 10, 'is_premonth'] = 1

        # 是否为月中
        total_dataset['is_midmonth'] = 0
        total_dataset.loc[(total_dataset['day'] > 10) & (
                total_dataset['day'] <= 20), 'is_midmonth'] = 1

        # 是否为月末
        total_dataset['is_tailmonth'] = 0
        total_dataset.loc[total_dataset['day'] > 20, 'is_tailmonth'] = 1

        # 是否为每月第一个周
        total_dataset['is_first_week'] = 0
        total_dataset.loc[total_dataset['week'] % 4 == 1, 'is_first_week'] = 1

        # 是否为每月第二个周
        total_dataset['is_second_week'] = 0
        total_dataset.loc[total_dataset['week'] % 4 == 2, 'is_second_week'] = 1

        # 是否为每月第三个周
        total_dataset['is_third_week'] = 0
        total_dataset.loc[total_dataset['week'] % 4 == 3, 'is_third_week'] = 1

        # 是否为每月第四个周
        total_dataset['is_fourth_week'] = 0
        total_dataset.loc[total_dataset['week'] % 4 == 0, 'is_fourth_week'] = 1

        return total_dataset.reset_index(drop=True)

    @staticmethod
    def window_evaluation_cv(df, target_col, ts_col, cv=5, window_length=30):
        """
        使用滑动窗口法切割时间序列数据，要求传入数据为时间顺序序列
        返回一个迭代器。
        """
        assert df.shape[0] >= (cv + 3) * window_length, "DataFrame length not enough to split."

        columns = [col for col in df.columns if col != target_col and col != ts_col
                   # and col != 'year'
                   ]

        start_index = df.shape[0] - cv * window_length

        for i in range(cv):
            # 每次切割
            training_set = df[columns][: start_index + i * window_length]
            train_target = df[target_col][: start_index + i * window_length]

            test_set = df[columns][start_index + i * window_length: start_index + (i + 1) * window_length]
            test_target = df[target_col][start_index + i * window_length: start_index + (i + 1) * window_length]

            yield training_set, train_target, test_set, test_target

