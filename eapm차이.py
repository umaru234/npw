# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 14:10:51 2025

@author: korrl
"""

import pandas as pd
import numpy as np

# ============================
# 1) 데이터 불러오기
# ============================
file_path = "정제한프본완.xlsx"  # 네가 쓰는 파일 이름으로 수정
df = pd.read_excel(file_path)

# ============================
# 2) 팀 구성 / EAPM 컬럼 정의
# ============================
ALLIES = ["gbr", "rus", "spa", "hre"]
REVOLT = ["fra", "usa", "tur"]

def eapm_col(nation):
    return f"{nation}_eapm"

# 각 팀의 eapm 컬럼 리스트
allies_eapm_cols = [eapm_col(n) for n in ALLIES]
revolt_eapm_cols = [eapm_col(n) for n in REVOLT]

# 컬럼 존재 여부 체크 (없으면 에러)
missing_allies = [c for c in allies_eapm_cols if c not in df.columns]
missing_revolt = [c for c in revolt_eapm_cols if c not in df.columns]
if missing_allies or missing_revolt:
    raise ValueError(f"EAPM 컬럼 누락: allies={missing_allies}, revolt={missing_revolt}")

# ============================
# 3) 국가별 eapm - 팀원 평균 eapm 차이 생성
#    diff = 본인 eapm - (팀원 eapm 평균)
# ============================
# 동맹팀
for nation in ALLIES:
    self_col = eapm_col(nation)
    teammate_cols = [eapm_col(n) for n in ALLIES if n != nation]

    # 팀원 평균 (NaN은 자동으로 무시됨)
    teammate_mean = df[teammate_cols].mean(axis=1)

    diff_col = f"{nation}_eapm_diff"
    df[diff_col] = df[self_col] - teammate_mean

# 혁명팀
for nation in REVOLT:
    self_col = eapm_col(nation)
    teammate_cols = [eapm_col(n) for n in REVOLT if n != nation]

    teammate_mean = df[teammate_cols].mean(axis=1)

    diff_col = f"{nation}_eapm_diff"
    df[diff_col] = df[self_col] - teammate_mean

# ============================
# 4) 컬럼 순서 조정:
#    각 XXX_eapm 바로 뒤에 XXX_eapm_diff 배치
# ============================
cols = list(df.columns)  # 현재 순서

for nation in ALLIES + REVOLT:
    base_col = eapm_col(nation)
    diff_col = f"{nation}_eapm_diff"

    if base_col not in cols or diff_col not in df.columns:
        # (해당 국가가 아예 없거나 diff가 없으면 스킵)
        continue

    # 기존 위치에서 diff 컬럼 제거 (뒤쪽에 붙어 있을 수 있음)
    if diff_col in cols:
        cols.remove(diff_col)

    # base_col 바로 뒤에 diff_col 삽입
    idx = cols.index(base_col)
    cols.insert(idx + 1, diff_col)

# 새로운 순서로 재정렬
df = df[cols]

# ============================
# 5) 저장
# ============================
out_path = "정제한프본완료.xlsx"
df.to_excel(out_path, index=False)
print("저장 완료:", out_path)
