from . import *


def is_in_dayrange(filepath, day_num):
    """
    文件是否在近day_num天内创建的，不含day_num当天
    """
    def timestamp2time(timestamp):
        timeStruct = datetime.datetime.fromtimestamp(timestamp)
        return timeStruct

    def get_file_modify_time(filepath):
        t = os.path.getmtime(filepath)
        return timestamp2time(t)

    return (get_file_modify_time(filepath) + datetime.timedelta(days=day_num)) > datetime.datetime.today()

def use_json(json_path, method='r', json_dict=None):
    if method == 'r':
        try:
            with open(json_path, 'r') as f:
                _ = json.load(f)
            return _

        except FileNotFoundError:
            raise FileNotFoundError(f"No such file or directory:{json_path}")

    elif method == 'w':
        if json_dict == None:
            raise ValueError("If method is 'w', the parameter json_dict must be not None.")
        with open(json_path, 'w') as f:
            json.dump(json_dict, f)

        return