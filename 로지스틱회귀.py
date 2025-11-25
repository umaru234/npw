# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 12:13:49 2025

@author: korrl
"""

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager, rc

#실습화일 목록에  포함된 malgun.ttf화일을 다운받아 'c:/Windows/fonts/' 폴더에 저장 
font_path = "c:/Windows/fonts/malgun.ttf" 

#font_path = "c:/test/malgun.ttf"
font_name = font_manager.FontProperties(fname = font_path).get_name()
matplotlib.rc('font', family=font_name)

 #마이너스(−) 기호가 깨지는 문제를 해결
matplotlib.rcParams['axes.unicode_minus'] = False  
#plt.title(), plt.xlabel(), plt.ylabel() 등에도 한글 폰트 적용.
plt.rcParams["font.family"] = font_name

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# =========================================
# 0. 데이터 불러오기
# =========================================
file_path = "정제한이벤트완료.xlsx"  # 필요하면 파일명 수정
df = pd.read_excel(file_path)

print("데이터 크기:", df.shape)

target_col = "is_a_win"

if target_col not in df.columns:
    raise ValueError(f"{target_col} 컬럼이 없습니다.")

# =========================================
# 1. 전략 행동 피처 자동 선택
#    - 수치형 컬럼 중에서
#      * 이벤트: *_decision, is_*_on  제외
#      * 특성: *_trait                제외
#      * 중점: *_focus                제외
#      * 보병수: *_inf_t*             제외
#      * 그 외 명백한 비-행동 변수들(duration, eapm, A_Reapm 등)도 제외
# =========================================

num_cols = df.select_dtypes(include=["number"]).columns.tolist()

exclude_exact = {
    "is_a_win",
    "is_r_win",
    "gameid",
    "duration",
    "A_Reapm",
    "allies_eapm_mean",
    "revolt_eapm_mean",
}

strategy_features = []

for col in num_cols:
    # 1) 타깃 & 명시적 제외 컬럼 스킵
    if col in exclude_exact:
        continue

    # 2) 이벤트 관련 컬럼 스킵
    if col.endswith("_decision"):          # warsaw_decision 등
        continue
    if col.startswith("is_") and col.endswith("_on"):  # is_warsaw_on 등
        continue

    # 3) 특성/중점 스킵
    if col.endswith("_trait"):   # gbr_trait 등
        continue
    if col.endswith("_focus"):   # gbr_focus 등
        continue

    # 4) 보병수(턴별 marine) 스킵
    if "inf_t" in col:           # gbr_inf_t1, usa_inf_t2 등
        continue

    # 5) eapm / cav_prod / 업그레이드 등도 제거해서
    #    "행동(logic) 변수"만 남기고 싶다면 추가로 필터
    if "eapm" in col:
        continue
    if "cav_prod" in col:
        continue
    if "art_prod" in col:
        continue
    if "battle" in col:
        continue
    if "_up_" in col:
        continue

    # 여기까지 살아남은 숫자 컬럼은 "전략행동/행동 카운트" 쪽일 확률이 높다
    strategy_features.append(col)

print("선택된 전략 행동 피처 수:", len(strategy_features))
print("예시 피처들:", strategy_features[:20])

# =========================================
# 2. X, y 구성 + 결측 처리
# =========================================
X = df[strategy_features].copy()
y = df[target_col].copy()

print("설명변수 결측 개수:\n", X.isna().sum())

# 결측이 있는 행은 일단 드롭
mask_notna = X.notna().all(axis=1) & y.notna()
X = X.loc[mask_notna].reset_index(drop=True)
y = y.loc[mask_notna].reset_index(drop=True)

print("결측 제거 후 데이터 크기:", X.shape)

y = y.astype(int)

# =========================================
# 3. 학습/검증 데이터 분리
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

print("Train 크기:", X_train.shape, " Test 크기:", X_test.shape)

# =========================================
# 4. 파이프라인 구성 (표준화 + 로지스틱 회귀)
# =========================================
logit_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=2000,
        penalty="l2",
        solver="lbfgs"
    )),
])

# =========================================
# 5. 학습
# =========================================
logit_pipe.fit(X_train, y_train)

# =========================================
# 6. 평가
# =========================================
y_train_pred = logit_pipe.predict(X_train)
y_test_pred  = logit_pipe.predict(X_test)

y_train_proba = logit_pipe.predict_proba(X_train)[:, 1]
y_test_proba  = logit_pipe.predict_proba(X_test)[:, 1]

print("\n=== Train 성능 ===")
print("Accuracy :", accuracy_score(y_train, y_train_pred))
print("ROC-AUC  :", roc_auc_score(y_train, y_train_proba))

print("\n=== Test 성능 ===")
print("Accuracy :", accuracy_score(y_test, y_test_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_test_proba))

print("\n=== 분류 리포트 (Test) ===")
print(classification_report(y_test, y_test_pred, digits=3))

print("\n=== 혼동 행렬 (Test) ===")
print(confusion_matrix(y_test, y_test_pred))

# =========================================
# 7. 계수(전략 행동 영향력) 보기
# =========================================
clf = logit_pipe.named_steps["clf"]
coef = clf.coef_[0]

coef_df = pd.DataFrame({
    "feature": strategy_features,
    "coef": coef,
})
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False)

print("\n=== 로지스틱 회귀 계수 (전략 행동만, 스케일 후 기준) ===")
print(coef_df[["feature", "coef"]])
