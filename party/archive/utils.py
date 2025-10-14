from datetime import datetime
import numpy as np
from typing import List
from collections import Counter, defaultdict

def flatten_list(nested_list: list) -> list:
    """  
    Flattens a nested list using recursion.  

    Args:  
        nested_list: The nested list to flatten.  

    Returns:  
        A flattened list.  
    """
    flattened = []
    for item in nested_list:
        if isinstance(item, list):  # Check if the item is a list
            # Recursively flatten the sublist
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

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

def print_dict_struct(dict_data, date_format='%Y-%m-%d', indent_level=-1):
    """  
    Print the structure of a dictionary.  

    This function traverses the input dictionary, printing out the keys and their corresponding values.   
    If a key is a valid date in the specified format, it will be skipped in the printing process.   
    It identifies NumPy arrays and prints their shapes, while also indicating the type of other values.  

    Args:  
        dict_data (dict): The dictionary to process and print.  
        date_format (str): The date format to check against (default is '%Y-%m-%d').  
        indent_level (int): The current indentation level for nested dictionaries (default is -1).  

    Returns:  
        None  
    """
    skip = False
    indent_level += 1

    for key, value in dict_data.items():
        if skip:
            continue

        if isinstance(value, dict):
            print(f"{' ' * 2*indent_level}- {key}")
            print_dict_struct(value, date_format,
                              indent_level)  # Recursive call
        else:
            if isinstance(value, np.ndarray):
                print(
                    f"{' ' * 2*indent_level}- {key}: A Numpy array with shape {value.shape}")
            elif isinstance(value, (int,float)):
                print(
                    f"{' ' * 2*indent_level}- {key}: {value}")
            else:
                print(f"{' ' * 2*indent_level}- {key}: {type(value)}")

        def is_like_n_format(num_str: str) -> bool:
            if num_str.startswith("n") and num_str[1:].isdigit():
                return True
            return False

        try: 
            # Check if the key is int
            if isinstance(key, int):
                skip = True
                continue
            # Check if the key is a neuron index
            if is_like_n_format(str(key)):
                skip = True
                continue
            # Check if the key is a valid date format
            datetime.strptime(str(key), date_format)
            skip = True
        except ValueError:
            skip = False