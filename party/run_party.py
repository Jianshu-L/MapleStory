# %%
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional
import pandas as pd
import numpy as np
from utils import flatten_list

# %%
@dataclass
class TeamSpec:
    # 用声明式“配方”定义每队所需角色，以及如何兜底
    # items 是按顺序的需求：("奶", 1) 表示抽取1人奶；("近战", 3) 表示抽3个近战
    main_character: Tuple[str, int]
    fallback: Optional[str] = None  # 可选：当主要类型空时，用次选类型补齐

def pop_random(lst: List, *, rng: random.Random) -> Optional[str]:
    if not lst:
        return None
    i = rng.randrange(len(lst))
    lst[i], lst[-1] = lst[-1], lst[i]
    return lst.pop()

def pop_order(lst: List) -> Optional[str]:
    if not lst:
        return None
    return lst.pop(0)

def form_team(spec: TeamSpec, buckets: Dict[str, List[str]], rng: random.Random, random: bool=False) -> Tuple[List[str], List[str], bool]:
    team_ids: List[str] = []
    team_jobs_type = []

    def take(role: str, count: int) -> int:
        taken = 0
        for _ in range(count):
            if random:
                pid = pop_random(buckets.get(role, []), rng)
            else:
                pid = pop_order(buckets.get(role, []))

            if pid is None:
                break
            team_ids.append(pid)
            taken += 1
        return taken

    # 先按主需求取
    role = spec.main_character[0]
    n = spec.main_character[1]
    got = take(role, n)
    team_jobs_type.append([role]*got)

    # 不够的从备选职业取
    if got < n and spec.fallback:
        fb_role = spec.fallback
        fb_got = take(fb_role, n-got)
        got += fb_got
        team_jobs_type.append([fb_role]*fb_got)

    # check number of member
    if got != n:
        full = False
    else:
        full = True

    return team_ids, team_jobs_type, full

import copy
def build_teams(roles_all, numbers_all, today_map, today_job_type, report, rng, num_member, team_number_start:int=0):
    """
    根据角色分配信息和权重表 today_map 组队。

    参数:
        roles_all: List[List[角色或角色列表]]   # 每个团队的角色信息
        numbers_all: List[List[int]]            # 每个团队对应的编号信息
        today_map: dict                         # 权重或匹配配置
        rng: Random-like 对象，用于 form_team()
        report: 报告对象，需有 add_warning() 方法
        num_member: int                         # 每队成员数量上限

    返回:
        (team_flatten, job_flatten)
        - team_flatten: 每个队伍合并后的队员ID列表
        - job_flatten: 每个队伍合并后的职业类型列表
    """
    team_flatten = []
    job_flatten = []

    # 深拷贝保证 today_map 不被修改
    buckets = copy.deepcopy(today_map)

    # 遍历每支队伍
    for i, (roles, numbers) in enumerate(zip(roles_all, numbers_all)):
        team_i, job_i = [], []

        # 遍历当前队伍所有角色
        for role, number in zip(roles, numbers):
            # 角色可能是 ["主职业","副职业"] 这种结构
            if isinstance(role, list):
                spec = TeamSpec(main_character=(role[0], number),
                                fallback=role[1])
            else:
                spec = TeamSpec(main_character=(role, number))

            # 生成队伍分配结果
            team_ids, team_jobs_type, full = form_team(spec, buckets, rng)

            # 如果成功组出角色，记录下来
            if team_ids:
                team_i.append(team_ids)
                job_i.append(team_jobs_type)

            # 如果资源（职业槽）没有填满
            if not full:
                report.add_warning(f"Group{i%2+1}Team{i//2+1+team_number_start} 缺 {role}")

        # 检查是否达标人数
        if (len(flatten_list(team_i)) < num_member) & (len(flatten_list(team_i)) > 0):
            report.add_warning(f"Group{i%2+1}Team{i//2+1+team_number_start} 人数不足")

        # 扁平化保存结果
        team_flatten.append(flatten_list(team_i))
        job_flatten.append(flatten_list(job_i))

    return team_flatten, job_flatten, buckets

# %%
import copy
from run import main

csv_name = "temp.csv"
today_map, today_job_type, report = main(csv_name)

random_seed = 2025
rng = random.Random(random_seed)
num_member = 6

# 1) 第一队:      奶1 + 火1 + 拳1 + 圣骑1 + 饺子1 + 需要拳的职业(弩，船)
# 2) 第二队(远程): 奶1 + 眼1 + (优先远程4, 不够用眼补齐至4)
# 3) 第三队(洗澡): 奶1 + 火1 + 刀
team_flatten = []
job_flatten = []
roles_all = [["奶", "火", "拳", "圣骑", "饺子", ["船", "弩"]],
            ["奶", "火", "拳", "圣骑", "饺子", ["弩", "船"]],
            ["奶", "火", "弓", ["标", "弓"]],
            ["奶", "火", "弓", ["标", "弓"]],
            ["奶", "火", ["刀", "饺子"]],
            ["奶", "火", ["刀", "饺子"]]]
numbers_all = [[1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 3],
            [1, 1, 1, 3],
            [1, 1, 4],
            [1, 1, 4]]
team_flatten, job_flatten, buckets = build_teams(roles_all, numbers_all, today_map, today_job_type, report, rng, num_member)

# 3) 第四队(剩余人数按远程近战分组)
id_remain = []
job_remain = []
job_type_remain = []
for k, v in buckets.items(): 
    if k != '单挂' and v:
        id_remain.extend(v)
        job_remain.extend([k]*len(v))
        job_type_remain.extend([today_job_type[v_i] for v_i in v])

remain_map: Dict[str, str] = {}
for i,id_i in enumerate(id_remain):
    remain_map[id_i] = job_remain[i]

buckets_remain: Dict[str, List[str]] = {}
for job_i in set(job_type_remain):
    for i,v in enumerate(job_type_remain):
        if v==job_i:
            if job_remain[i] not in buckets_remain:
                buckets_remain[job_remain[i]] = []
            buckets_remain[job_remain[i]].append(id_remain[i])

roles_all = [['奶', '近战'],['奶', '远程']]
numbers_all = [[1,5], [1,5]]
team_remain, job_remain, buckets_remain_remain = build_teams(roles_all, numbers_all, buckets_remain, today_job_type, report, rng, num_member, team_number_start=3)
team_flatten.extend(team_remain)
job_flatten.extend(job_remain)

for warning in report.warnings:
    print("Warnings:", warning)

print(buckets_remain_remain)

# %%
import string
from datetime import datetime
from openpyxl import load_workbook
letters = string.ascii_uppercase

# 1) 载入模板
wb = load_workbook("一条排班.xlsx")
ws = wb["Sheet1"]  # 替换为你的工作表名

# 2) 写入文件
now = datetime.now()

for i, (job, id_list) in enumerate(zip(job_flatten, team_flatten)):
    for j, job_i in enumerate(job):
        
        if i % 2 == 0:
            i_mod = i // 2
            ws[f'{letters[i_mod*2]}{3+j}'].value = id_list[j]
            ws[f'{letters[i_mod*2+1]}{3+j}'].value = job_i
        else:
            i_mod = i // 2
            ws[f'{letters[i_mod*2]}{12+j}'].value = id_list[j]
            ws[f'{letters[i_mod*2+1]}{12+j}'].value = job_i

ws['k2'] = '单挂'
for i, id_i in enumerate(buckets['单挂']):
    ws[f'k{2+i+1}'] = id_i

wb.save(f"{now.strftime('%Y%m%d')}一条排班.xlsx")

# %%
from tabulate import tabulate
# print
df = pd.read_excel(f"{now.strftime('%Y%m%d')}一条排班.xlsx", sheet_name="Sheet1", header=None)
df = df.fillna("")
num_members = (df != "").sum().sum()-df.shape[1]

print("====================")
print(f"Total Number of Members: {num_members}")
print(tabulate(df[1:], headers=df.iloc[0,:].values.astype('str'), tablefmt="github"))

# %%



