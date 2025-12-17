import json

import numpy as np
import pandas as pd

import re


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.datetime64):
            return str(obj)  # 将 datetime64 转换为字符串
        if isinstance(obj, np.dtype):
            return str(obj)  # 将 dtype 转换为字符串表示
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super(NpEncoder, self).default(obj)

def extract_exp_id(text, key="exp"):
    matcher = re.search(rf'<{key}>(.*?)</{key}>', text)
    result = matcher.group(1) if matcher else None
    return result
