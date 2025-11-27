# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 13:27:25 2025

@author: korrl
"""

import pandas as pd
import numpy as np
from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ============================================
# 0) 데이터 로드
# ============================================
file_path = "정제한표본추가7완.xlsx"
df = pd.read_excel(file_path)

nation_prefixes = ['gbr', 'fra', 'rus', 'tur', 'spa', 'hre', 'usa']
nation_prefixes_upper = [p.upper() for p in nation_prefixes]

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ============================================
# 1) 국가별 승리 타깃 정의
#    - 동맹: GBR, RUS, SPA, HRE → is_a_win(있으면) 또는 is_r_win == 0
#    - 혁명: FRA, USA, TUR → is_r_win == 1
# ============================================
def get_win_series(df, nation_prefix):
    nation = nation_prefix.upper()
    if nation in ['GBR', 'RUS', 'SPA', 'HRE']:
        if 'is_a_win' in df.columns:
            return df['is_a_win'].astype(int)
        else:
            return (df['is_r_win'] == 0).astype(int)
    else:
        return df['is_r_win'].astype(int)

# ============================================
# 2) 이진화 함수: 0/1이면 그대로, 그 외는 > median 기준
# ============================================
def binarize_series(series):
    vals = series.dropna().unique()
    # 이미 0/1 변수면 그대로
    if len(vals) > 0 and set(vals).issubset({0, 1}):
        return series.astype(bool)
    # 그 외에는 중앙값 기준
    med = series.median()
    if pd.isna(med):
        med = 0
    return series > med

# ============================================
# 3) 국가별 연관규칙 채굴 (eapm_diff 제외, min_support=0.10, max_len=3)
# ============================================
def mine_rules_for_nation(df, prefix, min_support=0.10, max_len=3, top_k=30):
    """
    prefix: 'gbr', 'fra', ...
    eapm_diff 포함 변수는 제외하고, 해당 국가 변수만으로
    WIN(그 국가 진영 승리) 규칙을 뽑는다.
    """
    cols = [c for c in num_cols
            if c.lower().startswith(prefix)
            and 'eapm_diff' not in c.lower()]
    if not cols:
        return pd.DataFrame(), None

    # 이진화
    bool_df = pd.DataFrame(index=df.index)
    for col in cols:
        series = df[col]
        bool_df[col] = binarize_series(series)
    bool_df = bool_df.fillna(False)

    # 승리 타깃
    win_series = get_win_series(df, prefix)
    bool_df['WIN'] = (win_series == 1).values

    items = [c for c in bool_df.columns if c != 'WIN']
    base_win_rate = bool_df['WIN'].mean()
    results = []

    for k in range(1, max_len + 1):
        for comb in combinations(items, k):
            mask_A = bool_df[list(comb)].all(axis=1)
            support_A = mask_A.mean()
            if support_A < min_support:
                continue
            support_A_and_win = (mask_A & bool_df['WIN']).mean()
            if support_A_and_win == 0:
                continue
            confidence = support_A_and_win / support_A
            lift = confidence / base_win_rate if base_win_rate > 0 else np.nan
            results.append({
                'antecedent': comb,
                'len_antecedent': k,
                'support_A': support_A,
                'support_A_and_win': support_A_and_win,
                'confidence': confidence,
                'lift': lift
            })

    rules_df = pd.DataFrame(results)
    if rules_df.empty:
        return pd.DataFrame(), base_win_rate

    rules_df_sorted = rules_df.sort_values(
        ['confidence', 'lift', 'support_A'],
        ascending=[False, False, False]
    )
    top_rules = rules_df_sorted.head(top_k).copy()
    # antecedent를 문자열로 저장 (나중에 다시 split)
    top_rules['antecedent'] = top_rules['antecedent'].apply(
        lambda x: ', '.join(x)
    )
    return top_rules, base_win_rate

nation_rule_results = {}
for prefix in nation_prefixes:
    rules, base_rate = mine_rules_for_nation(df, prefix, min_support=0.10, max_len=3, top_k=30)
    nation_rule_results[prefix.upper()] = (rules, base_rate)

# ============================================
# 4) 국가별로 의미 중복 규칙 제거 → 최대 6개 선택
#    - subset/superset, Jaccard 유사도 >= 0.67 제거
# ============================================
def select_nonredundant_rules(rules_df, max_rules=6):
    if rules_df is None or rules_df.empty:
        return rules_df

    selected_rows = []
    selected_sets = []

    def parse_set(ant_str):
        return set([a.strip() for a in ant_str.split(',') if a.strip()])

    for idx, row in rules_df.iterrows():
        cur_set = parse_set(row['antecedent'])
        if not cur_set:
            continue
        redundant = False
        for s in selected_sets:
            # subset/superset
            if cur_set.issubset(s) or cur_set.issuperset(s):
                redundant = True
                break
            # Jaccard
            inter = len(cur_set & s)
            union = len(cur_set | s)
            if union > 0:
                jaccard = inter / union
                if jaccard >= 0.67:
                    redundant = True
                    break
        if redundant:
            continue
        selected_rows.append(row)
        selected_sets.append(cur_set)
        if len(selected_rows) >= max_rules:
            break

    if not selected_rows:
        return rules_df.head(max_rules).copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)

selected_rules_by_nation = {}
for nation in nation_prefixes_upper:
    rules, base_rate = nation_rule_results.get(nation, (pd.DataFrame(), None))
    if rules is None or rules.empty:
        selected_rules_by_nation[nation] = (pd.DataFrame(), base_rate)
    else:
        rules_sorted = rules.sort_values(
            ['confidence', 'lift', 'support_A'],
            ascending=[False, False, False]
        )
        sel = select_nonredundant_rules(rules_sorted, max_rules=6)
        selected_rules_by_nation[nation] = (sel, base_rate)

# ============================================
# 5) 규칙 기반 피처 생성 (rule_*), 규칙 점수 = conf - base_win_rate
# ============================================
rule_feat_df = pd.DataFrame(index=df.index)
rule_feature_meta = []  # nation, col, score, antecedent_set, ...

for nation in nation_prefixes_upper:
    rules, base_rate = selected_rules_by_nation[nation]
    if rules is None or rules.empty or base_rate is None or base_rate == 0:
        continue
    for i, row in rules.iterrows():
        ant_str = row['antecedent']
        ant_items = [a.strip() for a in ant_str.split(',') if a.strip()]
        ant_set = set(ant_items)
        score = row['confidence'] - base_rate  # 규칙 점수
        if score <= 0:
            # 승률을 올리지 않는 규칙은 기여도에서 제외
            continue
        col_name = f"rule_{nation}_{i}"
        mask = pd.Series(True, index=df.index)
        for var in ant_items:
            if var not in df.columns:
                mask &= False
                break
            bin_ser = binarize_series(df[var])
            mask &= bin_ser.fillna(False)
        rule_feat_df[col_name] = mask.astype(int)
        rule_feature_meta.append({
            'nation': nation,
            'col': col_name,
            'score': float(score),
            'antecedent_set': ant_set,
            'confidence': float(row['confidence']),
            'base_win_rate': float(base_rate),
            'support_A': float(row['support_A'])
        })

# ============================================
# 6) 국가별 eapm_diff 가중치 생성
#     - GBR : 0.8 ~ 1.3
#     - RUS, FRA : 0.9 ~ 1.1
#     - 그 외 : 가중치 제거(1.0 고정)
# ============================================
eapm_diff_cols = {}
for nation in nation_prefixes_upper:
    pref = nation.lower()
    candidates = [c for c in df.columns
                  if pref in c.lower() and 'eapm_diff' in c.lower()]
    eapm_diff_cols[nation] = candidates[0] if candidates else None

weights = {}
for nation in nation_prefixes_upper:
    col = eapm_diff_cols.get(nation)
    # 가중치 제거 대상(GBR/RUS/FRA가 아니거나, 컬럼 자체 없음)은 1.0 고정
    if nation not in ['GBR', 'RUS', 'FRA'] or col is None or col not in df.columns:
        weights[nation] = pd.Series(1.0, index=df.index)
        continue

    s = df[col].astype(float)
    min_v = s.min()
    max_v = s.max()

    # min==max 이거나 NaN뿐이면 해당 국가는 EAPM 차이 정보가 의미 없으니 1.0 고정
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        weights[nation] = pd.Series(1.0, index=df.index)
        continue

    # 0~1로 min-max 정규화
    norm = (s - min_v) / (max_v - min_v)

    if nation == 'GBR':
        # 0.8 ~ 1.3 (폭 0.5)
        w = 0.8 + norm * 0.5
    elif nation in ['RUS', 'FRA']:
        # 0.9 ~ 1.1 (폭 0.2, 영향 좁게)
        w = 0.9 + norm * 0.2
    else:
        # 안전장치 (이론상 안 들어옴)
        w = pd.Series(1.0, index=df.index)

    w = w.fillna(1.0)
    weights[nation] = w


# ============================================
# 7) 규칙 점수 × 가중치 → 경기별 국가 raw 기여도
# ============================================
contrib_by_nation = {nation: pd.Series(0.0, index=df.index)
                     for nation in nation_prefixes_upper}

for meta in rule_feature_meta:
    nation = meta['nation']
    col = meta['col']
    score = meta['score']
    w = weights[nation]
    trig = rule_feat_df[col].astype(float)
    contrib = trig * score * w
    contrib_by_nation[nation] = contrib_by_nation[nation] + contrib

contrib_df = pd.DataFrame({
    f'contrib_{n}': s for n, s in contrib_by_nation.items()
})

# ============================================
# 8) 승리 진영 기준 aligned 기여도 (MVP용)
# ============================================
if 'is_r_win' not in df.columns:
    raise RuntimeError("is_r_win 컬럼이 필요합니다.")

is_r_win_series = df['is_r_win'].astype(int)

aligned_cols = {}
for nation in nation_prefixes_upper:
    side = 'allies' if nation in ['GBR', 'RUS', 'SPA', 'HRE'] else 'revolt'
    raw = contrib_by_nation[nation]
    aligned_vals = []
    for idx, val in raw.items():
        r_win = is_r_win_series.loc[idx]
        if (side == 'revolt' and r_win == 1) or (side == 'allies' and r_win == 0):
            aligned_vals.append(val)
        else:
            aligned_vals.append(-val)
    aligned_cols[f'aligned_{nation}'] = pd.Series(aligned_vals, index=df.index)

aligned_df = pd.DataFrame(aligned_cols)

# ============================================
# 9) 경기별 MVP (승리 팀 기준, 기존 규칙 유지)
# ============================================
mvp_nation_list = []
mvp_score_list = []

for idx in df.index:
    vals = {nation: aligned_df.loc[idx, f'aligned_{nation}']
            for nation in nation_prefixes_upper}
    items_sorted = sorted(vals.items(), key=lambda x: x[1], reverse=True)
    max_nation, max_val = items_sorted[0]
    second_val = items_sorted[1][1] if len(items_sorted) > 1 else float('-inf')
    gap = max_val - second_val
    if max_val > 0 and gap >= 0.1:
        mvp_nation_list.append(max_nation)
        mvp_score_list.append(max_val)
    else:
        mvp_nation_list.append(None)
        mvp_score_list.append(None)

mvp_series = pd.Series(mvp_nation_list, index=df.index, name='MVP_nation')
mvp_score_series = pd.Series(mvp_score_list, index=df.index, name='MVP_score')

# ============================================
# 10) 국가별 유저 컬럼 자동 탐색 (gbr_name, fra_name 등 추정)
# ============================================
user_cols = {}
for nation in nation_prefixes_upper:
    pref = nation.lower()
    candidates = [c for c in df.columns
                  if pref in c.lower()
                  and any(k in c.lower() for k in ['user', 'nick', 'name', 'id'])]
    user_cols[nation] = candidates[0] if candidates else None

# ============================================
# 11) 국가별 raw EAPM 컬럼 탐색 (eapm_diff 제외, mean_eapm 계산용)
# ============================================
eapm_cols_by_nation = {}
for nation in nation_prefixes_upper:
    pref = nation.lower()
    cands = [c for c in df.columns
             if pref in c.lower()
             and 'eapm_diff' not in c.lower()
             and 'eapm' in c.lower()]
    eapm_cols_by_nation[nation] = cands[0] if cands else None

# ============================================
# 12) 유저별 통계 (평균 aligned, 평균 EAPM)
# ============================================
user_stats = {}
for idx in df.index:
    for nation in nation_prefixes_upper:
        ucol = user_cols.get(nation)
        if not ucol or ucol not in df.columns:
            continue
        user = df.loc[idx, ucol]
        if pd.isna(user):
            continue

        aligned_val = aligned_df.loc[idx, f'aligned_{nation}']

        ecol = eapm_cols_by_nation.get(nation)
        eapm_val = df.loc[idx, ecol] if (ecol and ecol in df.columns) else np.nan

        if user not in user_stats:
            user_stats[user] = {
                'games': 0,
                'sum_aligned': 0.0,
                'sum_eapm': 0.0,
                'cnt_eapm': 0
            }

        user_stats[user]['games'] += 1
        user_stats[user]['sum_aligned'] += float(aligned_val)

        if not pd.isna(eapm_val):
            user_stats[user]['sum_eapm'] += float(eapm_val)
            user_stats[user]['cnt_eapm'] += 1

user_rows = []
for user, stat in user_stats.items():
    games = stat['games']
    sum_aligned = stat['sum_aligned']
    mean_aligned = sum_aligned / games if games > 0 else 0.0

    if stat['cnt_eapm'] > 0:
        mean_eapm = stat['sum_eapm'] / stat['cnt_eapm']
    else:
        mean_eapm = np.nan

    user_rows.append({
        'user': user,
        'games': games,
        'sum_aligned': sum_aligned,
        'mean_aligned': mean_aligned,
        'mean_eapm': mean_eapm
    })

user_df = pd.DataFrame(user_rows)

# ============================================
# 13) 개선된 Troll 정의:
#     - games >= 5
#     - mean_aligned 낮은 편 (하위 20%)
#     - mean_eapm도 낮은 편 (하위 30%)
# ============================================
min_games = 5
eligible_users = user_df[user_df['games'] >= min_games].copy()
eligible_users = eligible_users[~eligible_users['mean_eapm'].isna()].copy()

if not eligible_users.empty:
    contrib_q = eligible_users['mean_aligned'].quantile(0.20)
    eapm_q = eligible_users['mean_eapm'].quantile(0.30)

    troll_candidates = eligible_users[
        (eligible_users['mean_aligned'] <= contrib_q) &
        (eligible_users['mean_eapm']    <= eapm_q)
    ].copy()

    troll_df = troll_candidates.sort_values('mean_aligned', ascending=True).head(10)
else:
    troll_df = pd.DataFrame(columns=user_df.columns)

# ============================================
# 14) MVP 많이 먹은 유저 집계
# ============================================
mvp_user_counts = {}
for idx in df.index:
    mvp_nat = mvp_series.loc[idx]
    if mvp_nat is None:
        continue
    ucol = user_cols.get(mvp_nat)
    if not ucol or ucol not in df.columns:
        continue
    user = df.loc[idx, ucol]
    if pd.isna(user):
        continue
    mvp_user_counts[user] = mvp_user_counts.get(user, 0) + 1

mvp_user_rows = []
for user, cnt in mvp_user_counts.items():
    row = user_df[user_df['user'] == user]
    games = int(row['games'].iloc[0]) if not row.empty else 0
    mean_aligned = float(row['mean_aligned'].iloc[0]) if not row.empty else 0.0
    mvp_user_rows.append({
        'user': user,
        'mvp_games': cnt,
        'games': games,
        'mean_aligned': mean_aligned
    })

mvp_user_df = pd.DataFrame(mvp_user_rows).sort_values(
    'mvp_games', ascending=False
)

# ============================================
# 15) ACE: "진 팀에서 그나마 제일 잘한 사람"
#     - 진 팀 국가들의 raw contrib 중 최대값 국가 → ACE_nation
#     - 해당 국가 유저 → ACE_user
# ============================================
ace_nation_list = []
ace_score_list = []
ace_user_list = []

for idx in df.index:
    r_win = is_r_win_series.loc[idx]
    if r_win == 1:
        # 혁명 승 → 동맹이 패배
        losing_nations = ['GBR', 'RUS', 'SPA', 'HRE']
    else:
        # 동맹 승 → 혁명이 패배
        losing_nations = ['FRA', 'USA', 'TUR']

    best_nat = None
    best_val = None
    for nation in losing_nations:
        val = contrib_by_nation[nation].loc[idx]
        if (best_val is None) or (val > best_val):
            best_val = val
            best_nat = nation

    ace_nation_list.append(best_nat)
    ace_score_list.append(best_val)

    ucol = user_cols.get(best_nat)
    if ucol and ucol in df.columns:
        u = df.loc[idx, ucol]
        ace_user_list.append(u if pd.notna(u) else None)
    else:
        ace_user_list.append(None)

ace_nation_series = pd.Series(ace_nation_list, index=df.index, name='ACE_nation')
ace_score_series = pd.Series(ace_score_list, index=df.index, name='ACE_score')
ace_user_series = pd.Series(ace_user_list, index=df.index, name='ACE_user')

# 유저별 ACE 횟수
ace_user_counts = {}
for idx in df.index:
    user = ace_user_series.loc[idx]
    if user is None or (isinstance(user, float) and pd.isna(user)):
        continue
    ace_user_counts[user] = ace_user_counts.get(user, 0) + 1

ace_user_rows = []
for user, cnt in ace_user_counts.items():
    row = user_df[user_df['user'] == user]
    games = int(row['games'].iloc[0]) if not row.empty else 0
    mean_aligned = float(row['mean_aligned'].iloc[0]) if not row.empty else 0.0
    ace_user_rows.append({
        'user': user,
        'ace_games': cnt,
        'games': games,
        'mean_aligned': mean_aligned
    })

ace_user_df = pd.DataFrame(ace_user_rows).sort_values(
    'ace_games', ascending=False
)

# ============================================
# 16) (옵션) 규칙 기반 로지스틱 회귀 실행
#      - 패턴-only, 패턴+EAPMdiff
# ============================================
if 'is_r_win' in df.columns and not rule_feat_df.empty:
    y = df['is_r_win'].astype(int)

    # 패턴-only
    X_rule = rule_feat_df.copy()
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_rule, y, test_size=0.3, random_state=42, stratify=y
    )
    clf_rule = LogisticRegression(
        penalty='l2',
        solver='liblinear',
        C=1.0,
        max_iter=2000
    )
    clf_rule.fit(Xr_train, yr_train)
    yr_pred = clf_rule.predict(Xr_test)
    yr_proba = clf_rule.predict_proba(Xr_test)[:, 1]

    print("=== 패턴-only 로지스틱 (혁명 승) ===")
    print(classification_report(yr_test, yr_pred))
    print("Confusion matrix:\n", confusion_matrix(yr_test, yr_pred))
    print("ROC-AUC:", roc_auc_score(yr_test, yr_proba))

    coef_rule_df = pd.DataFrame({
        'feature': X_rule.columns,
        'coef': clf_rule.coef_[0]
    }).sort_values('coef', ascending=False)
    print("\n[패턴-only] 혁명 승률 올리는 규칙 Top 10")
    print(coef_rule_df.head(10))
    print("\n[패턴-only] 혁명 승률 깎는 규칙 Top 10")
    print(coef_rule_df.tail(10))

    # 패턴 + EAPMdiff
    eapm_diff_cols_all = [c for c in df.columns if 'eapm_diff' in c.lower()]
    eapm_diff_df = df[eapm_diff_cols_all].copy() if eapm_diff_cols_all else pd.DataFrame(index=df.index)

    X_mix = pd.concat([rule_feat_df, eapm_diff_df], axis=1)
    Xm_train, Xm_test, ym_train, ym_test = train_test_split(
        X_mix, y, test_size=0.3, random_state=42, stratify=y
    )
    clf_mix = LogisticRegression(
        penalty='l2',
        solver='liblinear',
        C=1.0,
        max_iter=2000
    )
    clf_mix.fit(Xm_train, ym_train)
    ym_pred = clf_mix.predict(Xm_test)
    ym_proba = clf_mix.predict_proba(Xm_test)[:, 1]

    print("\n=== 패턴+EAPMdiff 로지스틱 (혁명 승) ===")
    print(classification_report(ym_test, ym_pred))
    print("Confusion matrix:\n", confusion_matrix(ym_test, ym_pred))
    print("ROC-AUC:", roc_auc_score(ym_test, ym_proba))

    coef_mix_df = pd.DataFrame({
        'feature': X_mix.columns,
        'coef': clf_mix.coef_[0]
    }).sort_values('coef', ascending=False)
    print("\n[패턴+EAPMdiff] 혁명 승률 올리는 피처 Top 10")
    print(coef_mix_df.head(10))
    print("\n[패턴+EAPMdiff] 혁명 승률 깎는 피처 Top 10")
    print(coef_mix_df.tail(10))

# ============================================
# 17) 경기별 요약 DataFrame (원하면 CSV로 저장 가능)
# ============================================
summary_cols = ['gameid'] if 'gameid' in df.columns else []
summary = pd.concat(
    [df[summary_cols + ['is_r_win']] if summary_cols else df[['is_r_win']],
     contrib_df,
     aligned_df,
     mvp_series,
     mvp_score_series,
     ace_nation_series,
     ace_score_series,
     ace_user_series],
    axis=1
)

def show_selected_rules(selected_rules_by_nation, top=None, print_table=True):
    """
    selected_rules_by_nation에 들어있는 '선정 규칙(최대 6개)'들을
    하나의 DataFrame으로 합쳐서 반환.
    - top: 각 국가별로 상위 N개만 (보통 max 6이라 필요 없을 수도 있음)
    """
    rows = []
    for nation in sorted(selected_rules_by_nation.keys()):
        rules, base_rate = selected_rules_by_nation[nation]
        if rules is None or rules.empty:
            continue

        tmp = rules.copy()
        if top is not None:
            tmp = tmp.head(top)

        tmp.insert(0, 'nation', nation)
        tmp.insert(1, 'base_win_rate', base_rate)

        rows.append(tmp)

    if not rows:
        print("selected_rules_by_nation에 유효한 규칙이 없습니다.")
        return pd.DataFrame()

    sel_rules_df = pd.concat(rows, ignore_index=True)

    if print_table:
        print(sel_rules_df.to_string(index=False))

    return sel_rules_df

# 사용 예시:
selected_rules_df = show_selected_rules(selected_rules_by_nation, top=None)
# selected_rules_df.to_excel("selected_rules.xlsx", index=False)
selected_rules_df.to_excel("selected_rules.xlsx", index=False)

print("\n=== 경기별 요약 (상위 5행) ===")
print(summary.head())

print("\n=== MVP 상위 10명 ===")
print(mvp_user_df.head(10))

print("\n=== ACE (진 팀 최고) 상위 10명 ===")
print(ace_user_df.head(10))

print("\n=== Troll 후보 10명 (기여도↓ + EAPM↓) ===")
print(troll_df.head(10))
