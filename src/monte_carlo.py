
import copy
import numpy as np
from numpy import random
from dataclasses import dataclass, field
from src.utils import build_term_counts_dict, split_value_by_type
from typing import Callable, Dict, List, Tuple, Any, Optional

# ========== 物品类 ==========

@dataclass(frozen=True)
class Scroll:
    name: str
    success_p: float
    atk_value: int=0
    power_value: int=0
    ag_value: int=0
    int_value: int=0
    lucky_value: int=0
    destroy_on_fail_p: float = 0.0

@dataclass
class Item:
    name: str
    num_slots: int
    atk_value: int=0
    power_value: int=0
    ag_value: int=0
    int_value: int=0
    lucky_value: int=0
    attempts_used: int=0                                         # 砸过几次卷轴
    destroyed: bool = False                                      # 装备是否因卷轴消失
    history: List[str] = field(default_factory=list)             # 保存砸过的卷轴
    results_history: List[bool] = field(default_factory=list)    # 砸过的卷轴是否成功

    def __post_init__(self):
        if self.history is None:
            self.history = []
        if self.results_history is None:
            self.results_history = []

# ========== Monte Carlo ==========
from tqdm import tqdm
from collections import Counter
def monte_carlo_mix(
    rng: random.Generator,
    num_success_items: int,
    item_state: Item,
    Procedure: Callable,
    properties: List[str] = ["atk_value", "power_value", "ag_value", "int_value", "lucky_value"]
) -> Dict[str, Any]:

    items_distribution: List[int] = []
    scrolls: List[List[str]] = []
    destroys: List[List[bool]] = []
    success: List[List[int]] = []
    items_properties: Dict[str, List[List[int]]] = {
            f: [] for f in properties
    }

    def aggregate_batch(items_list: List[Item]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["scrolls_i"]  = [h for item in items_list for h in item.history]
        out["destroys_i"] = [item.destroyed for item in items_list]
        out["success_i"]  = [int(np.sum(item.results_history)) for item in items_list]
        for f in properties:
            out[f] = [getattr(item, f) for item in items_list]
        return out

    # 每次试验的类型计数
    for _ in tqdm(range(num_success_items), desc="Monte Carlo"):
        items_list = Procedure(rng, item_state)
        
        # statistics
        items_distribution.append(len(items_list))

        # 使用
        agg = aggregate_batch(items_list)

        scrolls.append(agg["scrolls_i"])
        destroys.append(agg["destroys_i"])
        success.append(agg["success_i"])

        for f in properties:
            items_properties[f].append(agg[f])

    properties_distribution: Dict[str, Dict] = {f: {} for f in properties}
    for f in properties:
        values_i = items_properties[f]
        destroyed_items, good_items = split_value_by_type(destroys, values_i)
        properties_distribution[f] = {
            "destroyed": destroyed_items,
            "good": good_items
        }

    return {
        "items_distribution": items_distribution,
        "scrolls_distribution": build_term_counts_dict(scrolls),
        "items_properties": properties_distribution,
        "destroy_distribution": destroys,
    }