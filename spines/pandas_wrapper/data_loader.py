from spines.built_in_exceptions.built_in_exceptions import FileLoadingError
from spines.pandas_wrapper.data_frame import DataFrameWrapper as dfw
import pandas as pd
import numpy as np


def load_data(path, file_type='csv', sheet_name=0, **kwargs) -> pd.DataFrame:
    """
    To load standard input data to machine memory.
    file_type: csv, excel
    """
    df = pd.DataFrame()
    if file_type == 'csv':
        try:
            df = pd.read_csv(path, **kwargs)

        except UnicodeError:
            df = pd.read_csv(path, encoding='gbk', **kwargs)

        except (FileNotFoundError, NameError):
            raise FileLoadingError(f"Please check your path, no such file or directory: '{path}'")
        except (RuntimeError, OSError):
            df = pd.read_csv(path, engine='python', **kwargs)

    elif file_type == 'excel':
        df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)

    return dfw(df)


def reduce_mem_data(df, verbose=True):
    """zip data."""
    start_mem = df.memory_usage().sum() / 1024 ** 2
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == int:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float64).min and c_max < np.finfo(
                        np.float64).max:
                    df[col] = df[col].astype(np.float64)
    if verbose:
        end_mem = df.memory_usage().sum() / 1024 ** 2
        print(
            f'Memory usage before optimization is: {round(start_mem, 2)} MB  ')
        print(f'Memory usage after optimization is: {round(end_mem, 2)} MB  ')
        print(
            f'Decreased by {round(100 * (start_mem - end_mem) / start_mem, 1)} %')
