import numpy as np
from typing import List
from collections import Counter, defaultdict

def build_term_counts_dict(data):
    n = len(data)  # 20000
    # 1) 先统计每个子列表的计数
    row_counters = [Counter(row) for row in data]
    # 2) 收集全局词表
    vocab = set()
    for cnt in row_counters:
        vocab.update(cnt.keys())
    # 3) 为每个词创建长度为 n 的列表，并按行填充
    result = {term: [0] * n for term in vocab}
    for i, cnt in enumerate(row_counters):
        for term, c in cnt.items():
            result[term][i] = c
    return result

def split_value_by_type(
    types_dict: List[List[bool]], 
    value_dict: List[List[int]]
    ):
    # 展平为 1D
    types = np.array([d for batch in types_dict for d in batch], dtype=bool)
    values = np.array([a for batch in value_dict for a in batch], dtype=int)

    # 掩码选择
    type_true_value = values[types]
    type_false_value = values[~types]

    return type_true_value, type_false_value
