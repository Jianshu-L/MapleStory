
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
            idx = state.attempts_used
            if idx >= len(seq):
                return None
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
                if scroll is None:
                    break

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

# ========== Example Running ==========
from src.monte_carlo import stop_all_success, stop_atk_at_least, strategy_fixed_sequence, monte_carlo_mix
from src.monte_carlo import stop_when_fail_or_consume_all_slots, stop_on_first_fail
import matplotlib.pyplot as plt
from src.plot import create_figure

mxb_to_rmb = 56 # 56W mxb = 1rmb
GROVE_ATTACK_PRICE = {
    "10": 1500,
    "11": 2500,
    "12": 5000,
    "13": 14000,
    "14": 700*mxb_to_rmb,
    "15": 2200*mxb_to_rmb,
    "16": 6000*mxb_to_rmb,
    "17": 12500*mxb_to_rmb
}

GROVE_SCROLL_PRICE = {
    "A": 33, 
    "B": 597, 
    "C": 14959, 
    "D": 2325
}
item_price = 2500

SCROLL_A = Scroll("A_10p_+3", success_p=0.11, atk_value=3, destroy_on_fail_p=0.0)
SCROLL_B = Scroll("B_60p_+2", success_p=0.66, atk_value=2, destroy_on_fail_p=0.0)
SCROLL_C = Scroll("C_30p_+3_boom50", success_p=0.33, atk_value=3, destroy_on_fail_p=0.50)
SCROLL_D = Scroll("D_70p_+2_boom50", success_p=0.77, atk_value=2, destroy_on_fail_p=0.50)

SCROLL_SET = {"A": SCROLL_A, "B": SCROLL_B, "C": SCROLL_C, "D": SCROLL_D}
NAME_TO_KEY = {SCROLL_A.name: "A", SCROLL_B.name: "B", SCROLL_C.name: "C", SCROLL_D.name: "D"}  # 便于反查类型键

def plot_bar_int(data, ax=None):
    values, counts = np.unique(data, return_counts=True)
    idx = values == 0
    values = np.delete(values, idx)
    counts = np.delete(counts, idx)
    if ax is None:
        plt.bar(values, counts/len(data), width=0.9, align='center', edgecolor='k')
        plt.xticks(values)  # 每个整数一个刻度
    else:
        ax.bar(values, counts/len(data), width=0.9, align='center', edgecolor='k')
        ax.set_xticks(values)  # 每个整数一个刻度

def plot_cdf_pdf(data, fig_title):

    num_items = []
    probability = []
    for i in range(0,np.max(data)+1):
        probability.append(np.sum(np.array(data) < i+1) / len(data))
        num_items.append(i+1)

    valid_num_items = range(0, int(np.mean(data) + 3*np.std(data, ddof=1)))
    valid_probability = np.array(probability)[valid_num_items]

    fig, axs = create_figure(1,3)
    fig.suptitle(fig_title)
    ax1 = fig.add_subplot(axs[0])
    ax1.plot(valid_probability)
    ax1.set_title("CDF")

    ax2 = fig.add_subplot(axs[1])
    ax2.hist(data, bins=np.arange(0,np.max(data),5))
    # ax2.axvline(np.max(valid_num_items), color='r')
    ax2.axvline(np.mean(data), color='r')
    ax2.set_title("PDF")
    ticks = ax2.get_yticks()
    ax2.set_yticks(ticks)  # 固定刻度位置，防止后续自动变化
    ax2.set_yticklabels([f"{t/len(data):g}" for t in ticks])

    ax3 = fig.add_subplot(axs[2])
    ax3.plot(np.diff(valid_probability))

def calculate_statistics(results):
    scrolls_dist = {}
    for scroll_name, scroll_distribution in results['scrolls_distribution'].items():
        scrolls_dist[scroll_name] = np.mean(scroll_distribution)

    num_groves = np.mean(results['items_distribution'])
    
    bad_groves_dist = {}
    v,c = np.unique(results['items_properties']['atk_value']['good'], return_counts=True)
    grove_sell = 0
    for v_i,c_i in zip(v,c):
        if v_i < np.max(v):
            bad_groves_dist[str(v_i)] = c_i

    return scrolls_dist, num_groves, bad_groves_dist

def main(scroll_list, item_state, item_price):
    import time
    seed = int(time.time())
    print(f"seed: {seed}")
    rng = np.random.default_rng(seed)
    trials = 100000

    target = stop_all_success()
    strat1 = strategy_fixed_sequence(scroll_list, SCROLL_SET)

    Procedure = stop_when_fail_or_consume_all_slots(strat1, target, stop_if_fail=True)
    # Procedure = stop_on_first_fail(strat1, target)
    # item_state = Item("BrownGrove", num_slots=7)
    results = monte_carlo_mix(rng, trials, item_state, Procedure, properties=["atk_value"])
    max_attack = np.max(results['items_properties']['atk_value']['good'])
    print("")

    scrolls_dist, num_groves, bad_groves_dist = calculate_statistics(results)
    print(f"Number of Groves cost: {num_groves}")

    scroll_cost = 0
    for key, value in scrolls_dist.items():
        scroll_i = GROVE_SCROLL_PRICE[NAME_TO_KEY[key]] * value
        scroll_cost += scroll_i
        print(f"Number of {key} cost: {value}")

    bad_grove_value = 0
    for key, value in bad_groves_dist.items():
        if key in GROVE_ATTACK_PRICE.keys():
            bad_grove_value += GROVE_ATTACK_PRICE[key] * value
    bad_grove_value /= trials
    print(f"bad grove value: {bad_grove_value}")

    total_cost = scroll_cost + num_groves*item_price
    print(f"Total Cost {total_cost}")
    # print(f"Actual Price {GROVE_ATTACK_PRICE[str(max_attack)]}")

    data = results['items_properties']['atk_value']['good']
    plot_bar_int(data)
    plt.title(scroll_list)