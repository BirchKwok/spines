import datetime
import json
import os


def is_in_day_range(filepath, day_num):
    """
    文件是否在近day_num天内创建的，不含day_num当天
    """
    def timestamp2time(timestamp):
        timeStruct = datetime.datetime.fromtimestamp(timestamp)
        return timeStruct

    def get_file_modify_time():
        t = os.path.getmtime(filepath)
        return timestamp2time(t)

    return (get_file_modify_time() + datetime.timedelta(days=day_num)) > datetime.datetime.today()


def use_json(json_path, method='r', json_dict=None):
    if method == 'r':
        try:
            with open(json_path, 'r') as f:
                _ = json.load(f)
            return _

        except FileNotFoundError:
            raise FileNotFoundError(f"No such file or directory:{json_path}")

    elif method == 'w':
        if json_dict is None:
            raise ValueError("If method is 'w', the parameter json_dict must be not None.")
        with open(json_path, 'w') as f:
            json.dump(json_dict, f, ensure_ascii=False)


def find_last_modify_file(directory_path):
    # 列出目录下所有的文件
    res_list = os.listdir(directory_path)
    # 对文件修改时间进行升序排列
    res_list.sort(key=lambda fn: os.path.getmtime(directory_path + '\\' + fn))
    # 获取最新修改时间的文件
    filetime = datetime.datetime.fromtimestamp(os.path.getmtime(directory_path + res_list[-1]))
    # 获取文件所在目录
    filepath = os.path.join(directory_path, res_list[-1])
    print("最新修改的文件(夹)：" + res_list[-1])
    print("时间：" + filetime.strftime('%Y-%m-%d %H-%M-%S'))
    return filepath


def order_file_by_modify_time(directory_path, reverse=True):
    """
    reverse: 升序True， 降序False
    """
    # 列出目录下所有的文件
    list_ = os.listdir(directory_path)
    # 对文件修改时间进行排列
    list_.sort(key=lambda fn: os.path.getmtime(directory_path + '\\' + fn), reverse=reverse)
    return list_
