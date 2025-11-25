# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 20:14:58 2025

@author: sgim4
"""
import pandas as pd
import numpy as np

# --- MICE(IterativeImputer) 불러오기 ---
from sklearn.experimental import enable_iterative_imputer  # 반드시 먼저 import
from sklearn.impute import IterativeImputer

# ========================================
# 1) 데이터 불러오기
# ========================================
file_path = "프본완.xlsx"   # 새 파일 이름
df_raw = pd.read_excel(file_path)

print("원본 행/열:", df_raw.shape)

if "duration" not in df_raw.columns:
    raise ValueError("duration 컬럼이 없습니다. 파일 구조를 확인하세요.")

# ========================================
# 1-1) 러시아/오스만 중점 결측치 처리
#      duration <= 400 이면 3으로 대체
# ========================================
if "rus_focus" in df_raw.columns:
    mask_rus = (df_raw["duration"] <= 400) & (df_raw["rus_focus"].isna())
    df_raw.loc[mask_rus, "rus_focus"] = 3
    print("rus_focus 3으로 대체된 개수:", mask_rus.sum())
else:
    print("rus_focus 컬럼이 없어 스킵합니다.")

if "tur_focus" in df_raw.columns:
    mask_tur = (df_raw["duration"] <= 400) & (df_raw["tur_focus"].isna())
    df_raw.loc[mask_tur, "tur_focus"] = 3
    print("tur_focus 3으로 대체된 개수:", mask_tur.sum())
else:
    print("tur_focus 컬럼이 없어 스킵합니다.")

# ========================================
# 2) duration > 150 인 경기만 사용
# ========================================
df = df_raw[df_raw["duration"] > 150].reset_index(drop=True)

print("필터링 전 행 수:", len(df_raw))
print("필터링 후 행 수:", len(df))

# ========================================
# 3) 변수 그룹 정의
# ========================================

# 특성(trait: 0/1 혹은 0/1/2)
trait_cols = [
    "gbr_trait", "usa_trait", "fra_trait",
    "rus_trait", "spa_trait", "hre_trait", "tur_trait",
]

# 중점(focus: 0/1, + 3(선택 전 종료) 등)
focus_cols = [
    "gbr_focus", "usa_focus", "fra_focus",
    "rus_focus", "spa_focus", "hre_focus", "tur_focus",
]

# 이벤트 decision (0/1, 미발동은 NaN)
event_cols = [
    "warsaw_decision", "canada_decision",
    "sweden_decision", "balkan_decision", "denmark_decision",
]

# 승리여부(0/1)
win_cols = ["is_a_win", "is_r_win"]

# 0/1 더미 변수들
# 프본완 기준: 이베리아 + 러시아전 + 파리침공까지 포함
dummy_cols = [
    "gbr_dropship_landing",
    "gbratkusa",
    "gbratktur",
    "fra_invasion",
    "fraatksam",
    "rusatkden",
    "usadefden",
    "rusconqnor",
    "rusfastinv",
    "rusnorinv",
    "spaatkusa",
    "gbrinvita",
    "usaatkcan",
    "usaatkmex",
    "turatkindia",
    "turatkegypt",
    "gbrdefspa",
    "fraatkpor",
    "fraatkrus",
    "isparisatked",
]

# 참고용: EAPM
eapm_cols = [
    "gbr_eapm", "usa_eapm", "fra_eapm",
    "rus_eapm", "spa_eapm", "hre_eapm", "tur_eapm",
]

# ========================================
# 4) 숫자형 컬럼만 선택
# ========================================
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
print("숫자형 컬럼 개수:", len(num_cols))

# ========================================
# 5) MICE 설정 및 적용
#    → event decision 컬럼과 gameid 는 imputation 대상에서 제외
#       (결측 그대로 유지)
# ========================================
exclude_from_impute = ["gameid"] + event_cols

impute_cols = [c for c in num_cols if c not in exclude_from_impute]

imputer = IterativeImputer(
    max_iter=20,
    random_state=42,
    sample_posterior=False,
)

numeric_data = df[impute_cols].values
numeric_imputed = imputer.fit_transform(numeric_data)

df_imputed = df.copy()
df_imputed[impute_cols] = numeric_imputed
# event decision, gameid 는 df 원본 값 그대로 유지 (결측 포함)

# ========================================
# 6) 범주형처럼 쓰는 숫자들:
#    trait, focus, event_decision, win, 더미들
#    → 반올림 + 원래 범위(min~max)로 클리핑
#    (event decision 은 MICE에서 건드리지 않아서, NaN 그대로 남음)
# ========================================
cat_like_cols = trait_cols + focus_cols + event_cols + win_cols + dummy_cols

for col in cat_like_cols:
    if col not in df_imputed.columns:
        continue

    original_non_na = df[col].dropna()
    if original_non_na.size == 0:
        ser = df_imputed[col].round().astype("Int64")
    else:
        min_val = original_non_na.min()
        max_val = original_non_na.max()

        ser = df_imputed[col].round()
        ser = ser.clip(lower=min_val, upper=max_val)
        ser = ser.astype("Int64")

    df_imputed[col] = ser

# ========================================
# 7) MICE 결과 중간 저장
# ========================================
output_path_mice = "프본완_cleaned_mice.xlsx"
df_imputed.to_excel(output_path_mice, index=False)
print("정제 + MICE 완료. 저장 위치:", output_path_mice)

# ========================================
# 8) 업그레이드 윈저라이즈
#    - 먼저 *_up_atk / *_up_def 가 없으면
#      세부 업그레이드로부터 생성
#    - FRA(프랑스): up_atk >= 14 → 13
#    - 그 외 국가: up_atk >= 10 → 9
#    - up_def > 5 → 5
# ========================================

# 8-0) 세부 업그레이드로부터 합계 up_atk / up_def 생성 (없을 때만)
nation_prefixes = ["gbr", "usa", "fra", "rus", "spa", "hre", "tur"]

has_up_atk = any(c.endswith("_up_atk") for c in df_imputed.columns)
has_up_def = any(c.endswith("_up_def") for c in df_imputed.columns)

if not (has_up_atk and has_up_def):
    for pref in nation_prefixes:
        atk_cols_n = [
            f"{pref}_up_tinf",
            f"{pref}_up_pgw",
            f"{pref}_up_zmelee",
            f"{pref}_up_zmissile",
        ]
        def_cols_n = [
            f"{pref}_up_tinf_def",
            f"{pref}_up_pga_def",
            f"{pref}_up_zcarapace",
        ]

        atk_cols_n = [c for c in atk_cols_n if c in df_imputed.columns]
        def_cols_n = [c for c in def_cols_n if c in df_imputed.columns]

        if atk_cols_n:
            df_imputed[f"{pref}_up_atk"] = df_imputed[atk_cols_n].sum(axis=1)
        if def_cols_n:
            df_imputed[f"{pref}_up_def"] = df_imputed[def_cols_n].sum(axis=1)

# 공/방 업 컬럼 자동 탐색
atk_cols = [c for c in df_imputed.columns if c.endswith("_up_atk")]
def_cols = [c for c in df_imputed.columns if c.endswith("_up_def")]
atk_def_cols = atk_cols + def_cols

print("공/방 업 컬럼:", atk_def_cols)

# 1) 국가별 윈저라이즈
for col in atk_def_cols:
    if col.startswith("fra_"):        # 프랑스
        df_imputed.loc[df_imputed[col] >= 14, col] = 13
    else:                             # 나머지 국가
        df_imputed.loc[df_imputed[col] >= 10, col] = 9

# 2) 방업은 최대 5로 제한
for col in def_cols:  # *_up_def 만 대상으로
    if col in df_imputed.columns:
        df_imputed.loc[df_imputed[col] > 5, col] = 5

# 정수형으로 통일
for col in atk_def_cols:
    df_imputed[col] = df_imputed[col].round().astype("Int64")

# 8-2) 국가별 공/방 합계 업 컬럼 제거 (*_up_atk, *_up_def)
if atk_def_cols:
    df_imputed.drop(columns=atk_def_cols, inplace=True)
    print("공/방 업 합계 컬럼 삭제 완료:", atk_def_cols)

# ========================================
# 8-1) 기병 생산 상한: 최대 20으로 제한
#      (예: gbr_cav_prod, rus_cav_prod, ... )
# ========================================
cav_cols = [c for c in df_imputed.columns if "cav_prod" in c]

print("기병 생산 컬럼:", cav_cols)

for col in cav_cols:
    df_imputed.loc[df_imputed[col] > 20, col] = 20
    df_imputed[col] = df_imputed[col].round().astype("Int64")

# ========================================
# 8-2) A_Reapm 파생변수 생성
#      A_Reapm = (동맹 평균 eapm) - (혁명 평균 eapm)
# ========================================
ALLIES_EAPM_COLS = ["gbr_eapm", "rus_eapm", "spa_eapm", "hre_eapm"]
REVOLT_EAPM_COLS = ["fra_eapm", "usa_eapm", "tur_eapm"]

df_imputed["allies_eapm_mean"] = df_imputed[ALLIES_EAPM_COLS].mean(axis=1)
df_imputed["revolt_eapm_mean"] = df_imputed[REVOLT_EAPM_COLS].mean(axis=1)
df_imputed["A_Reapm"] = df_imputed["allies_eapm_mean"] - df_imputed["revolt_eapm_mean"]

# ========================================
# 9) 최종 저장 (MICE + 윈저라이즈 + A_Reapm)
# ========================================
output_path_winsor = "정제한프본완.xlsx"
df_imputed.to_excel(output_path_winsor, index=False)
print("윈저라이즈 완료, 저장:", output_path_winsor)
