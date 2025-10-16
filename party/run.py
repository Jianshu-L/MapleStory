from __future__ import annotations
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional
import pandas as pd
import numpy as np
from utils import flatten_list

# ---------- Helpers ----------

def clean_id(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def find_contain_index(data: list, target: str):
    return [i for i,j in enumerate(data) if j in target]

def pop_random(lst: List, *, rng: random.Random) -> Optional[str]:
    if not lst:
        return None
    i = rng.randrange(len(lst))
    lst[i], lst[-1] = lst[-1], lst[i]
    return lst.pop()

# ---------- Mapping inputs to jobs ----------

@dataclass
class MappingSource:
    # 基础数据源：并行数组或从外部注入的主数据
    ids: List[str]          # 主数据中的 id 列表（去重后）
    jobs: List[str]         # 对应岗位（可选）
    job_types: List[str]    # 对应职业标签，如 "奶" "拳" "眼" "火" "近战" "远程"

def build_index(ms: MappingSource) -> Dict[str, Tuple[str, str]]:
    """
    将主数据 ids → (job, job_type) 的唯一映射构建出来。
    """
    df = {}
    for i, pid in enumerate(ms.ids):
        pid_clean = clean_id(pid)
        if not pid_clean:
            continue
        if pid_clean in df:
            # 如有重复，raise warning and pass
            Warning(f"Dulicate Id {pid_clean} in repo, ignore")
            continue

        job = ms.jobs[i]
        job_type = ms.job_types[i]
        df[pid_clean] = (job, job_type)
    return df

# ---------- Reporting ----------

@dataclass
class GroupReport:
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    unmapped_ids: List[str] = field(default_factory=list)   # CSV 中找不到映射的 ID
    invalid_ids: List[str] = field(default_factory=list)    # 空/非法/重复 ID
    leftover_buckets: Dict[str, List[str]] = field(default_factory=dict)  # 分组后仍剩余的人
    grouped: List[List[str]] = field(default_factory=list)  # 各队成员 ID
    grouped_jobs: List[List[str]] = field(default_factory=list)  # 各队成员的 job_type
    ok: bool = True

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

# ---------- Core pipeline ----------
def handle_find_contain_index_bug(pid: str, repo_id: List[str], index_list: List[int], report: GroupReport):
    possible_ids = [repo_id[i] for i in index_list]
    sorted_possible_ids = sorted(possible_ids, key=len, reverse=True)
    report.invalid_ids.append((pid, sorted_possible_ids))
    report.add_warning(f"duplicated matching in Repo: {pid} {sorted_possible_ids}")

    argmax = np.argmax([len(w) for w in possible_ids])
    index_list[0] = index_list[argmax]
    return index_list

def get_job_from_pid(today_pids: list, index_map: Dict[str, Tuple[str, str]]):
    repo_id = list(index_map.keys())
    repo_jobs = [i[0] for i in index_map.values()]

    buckets_job = []
    for pid in today_pids:
        index_list = find_contain_index(repo_id, pid)
        if index_list:
            if len(index_list) != 1:
                raise ValueError("Something Wrong")
            buckets_job.append(repo_jobs[index_list[0]])
        else:
            buckets_job.append("")
        
    return buckets_job

def map_today_ids(csv_path: str, index_map: Dict[str, Tuple[str, str]], report: GroupReport) -> Dict[str, str]:
    """
    读取 CSV, 按 index_map 将 today 的 id → job_type。
    找不到映射的, 记为 unmapped_ids; 空值记 invalid_ids; 映射不到 job_type 的给空字符串。
    """
    df = pd.read_csv(csv_path, header=None)
    raw_ids = np.squeeze(df.values).tolist()
    today_ids = [clean_id(x) for x in raw_ids]

    today_map: Dict[str, str] = {}
    seen = set()

    for pid in today_ids:
        if not pid:
            report.invalid_ids.append(pid)
            continue
        if pid in seen:
            report.invalid_ids.append(pid)
            report.add_warning(f"duplicated id in CSV: {pid}")
            continue

        seen.add(pid)
        
        if "单挂" in pid:
            report.unmapped_ids.append(pid)
            today_map[pid] = ""
            continue

        repo_id = list(index_map.keys())
        index_list = find_contain_index(repo_id, pid)
        if index_list:
            if len(index_list) != 1:
                index_list = handle_find_contain_index_bug(pid, repo_id, index_list, report)

            job, job_type = index_map[repo_id[index_list[0]]]
            today_map[repo_id[index_list[0]]] = job or ""
        else:
            report.unmapped_ids.append(pid)
            today_map[pid] = ""  # 未映射先留空，后续会进入“未知”类

    # 人数校验（可选）
    if len(today_ids) != df.shape[0]:
        report.add_warning("CSV squeeze produced different length; check data shape.")

    return today_map

def bucket_by_job_type(today_map: Dict[str, str]) -> Dict[str, List[str]]:
    buckets = defaultdict(list)
    for pid, jt in today_map.items():
        key = jt.strip() if isinstance(jt, str) else ""
        buckets[key].append(pid)
    return buckets

@dataclass
class TeamSpec:
    # 用声明式“配方”定义每队所需角色，以及如何兜底
    # items 是按顺序的需求：("奶", 1) 表示抽取1人奶；("近战", 3) 表示抽3个近战
    main_character: Tuple[str, int]
    fallback: Optional[str] = None  # 可选：当主要类型空时，用次选类型补齐

def form_team(spec: TeamSpec, buckets: Dict[str, List[str]], rng: random.Random) -> Tuple[List[str], List[str], bool]:
    team_ids: List[str] = []
    # team_jobs_type: List[str] = []

    def take(role: str, count: int) -> int:
        taken = 0
        for _ in range(count):
            pid = pop_random(buckets.get(role, []), rng=rng)
            if pid is None:
                break
            team_ids.append(pid)
            # team_jobs_type.append(role)
            taken += 1
        return taken

    # 先按主需求取
    role = spec.main_character[0]
    n = spec.main_character[1]
    got = take(role, n)
    # 不够的从备选职业取
    if got < n and spec.fallback:
        fb_role = spec.fallback
        fb_got = take(fb_role, n-got)
        got += fb_got

    # check number of member
    if got != n:
        full = False
    else:
        full = True

    # team_jobs
    team_jobs = get_job_from_pid(team_ids, index_map)
    return team_ids, team_jobs, full

def form_teams(buckets: Dict[str, List[str]], report: GroupReport, rng: random.Random) -> Tuple[List[List[str]], List[List[str]]]:
    """
    将你原先的四队策略以更清晰的方式实现：
    1) 第一队: 奶1 + 拳1 + 火1 + 船1 + 近战2
    2) 第二队: 奶1 + 眼1 + (优先远程4, 不够用眼补齐至4)
    3) 第三队: 奶1 + 火1 + 近战
    4) 第四队: CSV 中 today_map 但未被前3队使用的(含未知/未映射)
    """
    num_member = 6

    # 1) 第一队 奶+火+拳+圣骑+战士（洗澡）
    team1, job1 = [], []
    spec = TeamSpec(main_character=("奶", 1))
    team_ids, team_jobs, full = form_team(spec, buckets, rng)
    team1.append(team_ids)
    job1.append(team_jobs)
    if not full:
        report.add_warning(f"Team1缺奶")

    spec = TeamSpec(main_character=("拳", 1))
    team_ids, team_jobs, full = form_team(spec, buckets, rng)
    team1.append(team_ids)
    job1.append(team_jobs)
    if not full:
        report.add_warning(f"Team1缺拳")

    spec = TeamSpec(main_character=("火", 1))
    team_ids, team_jobs, full = form_team(spec, buckets, rng)
    team1.append(team_ids)
    job1.append(team_jobs)
    if not full:
        report.add_warning(f"Team1缺火")

    n = num_member - len(team1)
    spec = TeamSpec(main_character=("战士", n))
    team_ids, team_jobs, full = form_team(spec, buckets, rng)
    team1.append(team_ids)
    job1.append(team_jobs)
    if not full:
        report.add_warning(f"Team1人不满")

    team1_flatten = flatten_list(team1)
    job1_flatten = flatten_list(job1)

    # 2) 第二队: 奶 + 眼 + 远程（船，标等）
    team2, job2 = [], []
    spec = TeamSpec(main_character=("奶", 1))
    team_ids, team_jobs, full = form_team(spec, buckets, rng)
    team2.append(team_ids)
    job2.append(team_jobs)
    if not full:
        report.add_warning(f"Team2缺奶")

    spec = TeamSpec(main_character=("眼", 1))
    team_ids, team_jobs, full = form_team(spec, buckets, rng)
    team2.append(team_ids)
    job2.append(team_jobs)
    if not full:
        report.add_warning(f"Team2缺眼")

    n = num_member - len(team2)
    spec = TeamSpec(main_character=("远程", n), fallback="眼")
    team_ids, team_jobs, full = form_team(spec, buckets, rng)
    team2.append(team_ids)
    job2.append(team_jobs)
    if not full:
        report.add_warning(f"Team2远程人不满")

    team2_flatten = flatten_list(team2)
    job2_flatten = flatten_list(job2)

    # 3) 第三队: 奶+火+远程小脆皮+其他
    team3, job3 = [], []
    pid = pop_random(buckets.get("奶", []), rng=rng)
    if pid is None:
        report.add_warning("Team3缺奶")
    else:
        team3.append(pid); job3.append("奶")

    pid = pop_random(buckets.get("火", []), rng=rng)
    if pid is None:
        report.add_warning("Team3缺火")
    else:
        team3.append(pid); job3.append("火")
    
    # 输出
    n = num_member - len(team3)
    spec = TeamSpec(main_character=("远程", n), fallback="眼")
    team_ids, team_jobs, full = form_team(spec, buckets, rng)
    team3.append(team_ids)
    job3.append(team_jobs)
    if not full:
        report.add_warning(f"Team3输出人不满")

    team3_flatten = flatten_list(team3)
    job3_flatten = flatten_list(job3)

    # 4) 第四队: 奶 + 刀 + 火
    team4, job4 = [], []
    pid = pop_random(buckets.get("奶", []), rng=rng)
    if pid is None:
        report.add_warning("Team4缺奶")
    else:
        team4.append(pid); job4.append("奶")

    # 刀+火
    n = num_member - len(team4)
    spec = TeamSpec(main_character=("刀", n), fallback="火")
    team_ids, team_jobs, full = form_team(spec, buckets, rng)
    team4.append(team_ids)
    job4.append(team_jobs)
    if not full:
        report.add_warning(f"Team4输出人不满")

    # 剩余输出
    n = num_member - len(team4)
    spec = TeamSpec(main_character=("刀", n), fallback="战士")
    team_ids, team_jobs, full = form_team(spec, buckets, rng)
    team4.append(team_ids)
    job4.append(team_jobs)
    if not full:
        report.add_warning(f"Team4输出人不满")

    team4_flatten = flatten_list(team4)
    job4_flatten = flatten_list(job4)

    # 5) 第五队: 奶1 + 单挂 + 多出的人
    team5, job5 = [], []
    pid = pop_random(buckets.get("奶", []), rng=rng)
    if pid is None:
        report.add_warning("Team5缺奶")
    else:
        team5.append(pid); job5.append("奶")

    # 将剩余所有人塞进5队
    for role, ids in buckets.items():
        while ids:
            team5.append(ids.pop())
            job5.append(role)

    team5_flatten = flatten_list(team5)
    job5_flatten = flatten_list(job5)

    return [team1_flatten, team2_flatten, team3_flatten, team4_flatten, team5_flatten], [job1_flatten, job2_flatten, job3_flatten, job4_flatten, job5_flatten]

def run(csv_path: str, ms: MappingSource, *, random_seed: int = 2025) -> GroupReport:
    rng = random.Random(random_seed)
    report = GroupReport()

    # index_map = build_index(ms)

    today_map = map_today_ids(csv_path, index_map, report)
    buckets = bucket_by_job_type(today_map)

    # 统计映射成功的人
    mapped_ids = set(pid for pid, jt in today_map.items() if jt)
    unknown_ids = [pid for pid, jt in today_map.items() if not jt]
    if unknown_ids:
        report.add_warning(f"{len(unknown_ids)} IDs have empty/unknown job_type.")

    grouped, grouped_jobs = form_teams(buckets, report, rng)

    # 统计余量（如果任何桶还有剩余，说明第3队没吃完所有人）
    leftover = {role: ids[:] for role, ids in buckets.items() if ids}
    if leftover:
        report.leftover_buckets = leftover
        report.add_warning(f"Leftover after forming teams: { {k: len(v) for k,v in leftover.items()} }")

    report.grouped = grouped
    report.grouped_jobs = grouped_jobs

    # 最终人数校验：成功分组 + 单挂 应等于 CSV 读入有效行数
    grouped_count = sum(len(t) for t in grouped)
    csv_count = pd.read_csv(csv_path, header=None).shape[0]
    if grouped_count != csv_count:
        report.add_warning(f"Headcount mismatch: grouped={grouped_count}, csv={csv_count}")

    return report

def get_key_by_value(d, value):
    return [k for k, v in d.items() if v == value]

from openpyxl import load_workbook
from datetime import datetime
import string
from tabulate import tabulate
if __name__ == "__main__":
    df = pd.read_csv("repo.csv")
    id = df.id.tolist()
    job = df.job.tolist()
    job_type = df.job_type.tolist()

    ms = MappingSource(
    ids=id,                # 你的主数据 id 列表
    jobs=job,              # 可不使用但可保留
    job_types=job_type,    # 与 id 对齐的职业类型
    )
    index_map = build_index(ms)

    report = GroupReport()
    csv_path = "temp.csv"
    today_maps = map_today_ids(csv_path, index_map, report)

    # 问题汇总
    print("Warnings:", report.warnings)
    print("Errors:", report.errors)
    print("Unmapped:", report.unmapped_ids)
    print("Invalid:", report.invalid_ids)
    print("Leftover:", report.leftover_buckets)

    today_members = {}
    for job_i in np.unique(list(today_maps.values())):
        if not job_i:
            key = "单挂"
        else:
            key = str(job_i)

        today_members[key] = get_key_by_value(today_maps, job_i)

    letters = string.ascii_uppercase
    job_order = ["奶", "火", '圣骑', '拳', '弩', '船', '饺子', '刀']

    # 1) 载入模板
    wb = load_workbook("一条分组.xlsx")
    ws = wb["Sheet1"]  # 替换为你的工作表名

    # 2) 写入文件
    now = datetime.now()

    last_index = len(job_order)-1
    for job, id_list in today_members.items():
        if job == "单挂":
            continue

        if job in job_order:
            i = job_order.index(job)
        else:
            last_index += 1
            i = last_index

        ws[f'{letters[i]}1'].value = job

        for id_i, id in enumerate(id_list):
            ws[f'{letters[i]}{id_i+2}'].value = id

    job = "单挂"
    ws[f'{letters[last_index+1]}1'].value = job

    id_list = today_members[job]
    for id_i, id in enumerate(id_list):
        ws[f'{letters[last_index+1]}{id_i+2}'].value = id

    wb.save(f"{now.strftime('%Y%m%d')}一条.xlsx")
    
    # print
    df = pd.read_excel(f"{now.strftime('%Y%m%d')}一条.xlsx", sheet_name="Sheet1", header=None)
    df = df.fillna("")
    num_members = (df != "").sum().sum()-df.shape[1]

    df_raw = pd.read_csv(csv_path, header=None)
    if num_members != df_raw.shape[0]:
        Warning("Final excel sheet produced different length; check data shape.")

    print("====================")
    print(f"Total Number of Members: {num_members}")
    print(tabulate(df[1:], headers=df.iloc[0,:], tablefmt="github"))
