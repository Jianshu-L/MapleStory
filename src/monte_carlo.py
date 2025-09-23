
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

Strategy = Callable[[Item], Scroll]
StopCondition = Callable[[Item], bool]
# ========== 策略工厂 ==========
def strategy_fixed_sequence(
    seq: List[str],
    SCROLL_SET: Dict[str, Scroll]
    ) -> Strategy:
    """
    固定序列：按照提供的卷轴名序列循环使用（长度不足时可循环或截断）。
    例如：["C", "C", "B", "B", "B", "B", "B"]
    """

    scrolls = [SCROLL_SET[name] for name in seq]

    def strat(state: Item) -> Scroll:
        if state.attempts_used < state.num_slots:
            idx = min(state.attempts_used, len(seq) - 1)
            return scrolls[idx]
        else:
            raise ValueError("No more slots to update")
    return strat

# ========== 目标工厂 ==========
def stop_all_success() -> StopCondition:
    # 目标：在不爆装情况下，全成功
    def cond(state: Item) -> bool:
        return (np.sum(state.results_history) == state.num_slots) and (not state.destroyed)
    return cond

def stop_atk_at_least(target_atk: int) -> StopCondition:
    # 目标：总攻击力达到阈值（不爆装）
    def cond(state: Item) -> bool:
        return (state.atk_value >= target_atk) and (not state.destroyed)
    return cond

# ========== 流程工厂 ==========
def stop_when_fail_or_consume_all_slots(
    strategy: Strategy,
    stop_condition: StopCondition,
    stop_if_fail: bool = False, # 是否失败就不再砸卷
) -> list:

    def run_procedure(rng: np.random.Generator,
                    item_template: Item,
    ):
        all_items_list = []

        while True:

            item_state = copy.deepcopy(item_template)
            # 对每件装备的砸卷处理
            while item_state.attempts_used < item_state.num_slots:
                scroll = strategy(item_state)

                # 记录本次使用的卷轴
                item_state.history.append(scroll.name)

                # 消耗一次尝试
                item_state.attempts_used += 1

                # 判定
                if rng.random() < scroll.success_p:
                    item_state.atk_value += scroll.atk_value
                    item_state.power_value += scroll.power_value
                    item_state.ag_value += scroll.ag_value
                    item_state.int_value += scroll.int_value
                    item_state.lucky_value += scroll.lucky_value

                    item_state.results_history.append(True)
                else:
                    item_state.results_history.append(False)

                    # 判定该装备是否消失
                    if scroll.destroy_on_fail_p > 0 and (rng.random() < scroll.destroy_on_fail_p):
                        item_state.destroyed = True
                        break
                    
                    # 新策略：只要失败，就不再继续砸
                    if stop_if_fail:
                        break

            all_items_list.append(item_state)
            # 完成该装备的一轮砸卷后
            if stop_condition(item_state):
                return all_items_list
    
    return run_procedure

def stop_on_first_fail(
    strategy: Strategy,
    stop_condition: StopCondition,
) -> list:

    def run_procedure(rng: np.random.Generator,
                    item_template: Item,
    ):
        all_items_list = []

        while True:

            item_state = copy.deepcopy(item_template)
            # 对每件装备的砸卷处理
            while item_state.attempts_used < item_state.num_slots:
                scroll = strategy(item_state)

                # 记录本次使用的卷轴
                item_state.history.append(scroll.name)

                # 消耗一次尝试
                item_state.attempts_used += 1

                # 判定
                if rng.random() < scroll.success_p:
                    item_state.atk_value += scroll.atk_value
                    item_state.power_value += scroll.power_value
                    item_state.ag_value += scroll.ag_value
                    item_state.int_value += scroll.int_value
                    item_state.lucky_value += scroll.lucky_value

                    item_state.results_history.append(True)
                else:
                    item_state.results_history.append(False)

                    # 判定该装备是否消失
                    if scroll.destroy_on_fail_p > 0 and (rng.random() < scroll.destroy_on_fail_p):
                        item_state.destroyed = True
                        break
                    
                    # 第一次失败就不砸了
                    if item_state.attempts_used == 1:
                        break

            all_items_list.append(item_state)
            # 完成该装备的一轮砸卷后
            if stop_condition(item_state):
                return all_items_list
    
    return run_procedure