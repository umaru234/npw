# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 12:34:32 2025

@author: korrl
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# =========================================
# 0. 데이터 불러오기
# =========================================
file_path = "정제한이벤트완료.xlsx"  # 너가 쓰던 최신 정제본
df = pd.read_excel(file_path)

print("데이터 크기:", df.shape)

target_col = "is_a_win"

if target_col not in df.columns:
    raise ValueError(f"{target_col} 컬럼이 없습니다.")

# =========================================
# 1. 피처 정의 (이전 main 로지스틱과 동일)
# =========================================

core_features = [
    # 경기 특성
    "duration",
    "A_Reapm",

    # 영국 핵심 행동
    "gbr_dropship_landing",
    "gbratktur",
    "gbrinvita",

    # 러시아 핵심 행동
    "rusfastinv",
    "rusconqnor",

    # 혁명 핵심 행동
    "fra_invasion",
    "turatkegypt",

    # 스페인 미국 공격
    "spaatkusa",

    # 이벤트 발동 여부
    "is_warsaw_on",
    "is_canada_on",
    "is_sweden_on",
]

trait_add = [
    "gbr_trait",
    "fra_trait",
    "rus_trait",
    "tur_trait",
    "spa_trait",
    "hre_trait",
]

focus_add = [
    "gbr_focus",
    "fra_focus",
    "rus_focus",
    "tur_focus",
    "spa_focus",
    "hre_focus",
]

inf_add = [
    "gbr_inf_t1",
    "usa_inf_t2",
    "tur_inf_t2",
    "fra_inf_t3",
]

features = core_features + trait_add + focus_add + inf_add

print("사용 피처 개수:", len(features))

missing = [c for c in features + [target_col] if c not in df.columns]
if missing:
    raise ValueError(f"다음 컬럼이 데이터에 없습니다: {missing}")

# =========================================
# 2. X, y 구성 + 결측 제거
# =========================================
X = df[features].copy()
y = df[target_col].copy()

print("결측 개수:\n", X.isna().sum())

mask_notna = X.notna().all(axis=1) & y.notna()
X = X.loc[mask_notna].reset_index(drop=True)
y = y.loc[mask_notna].reset_index(drop=True)

print("결측 제거 후 데이터 크기:", X.shape)

y = y.astype(int)

# =========================================
# 3. Train / Test 분할
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
# 4. 디시전 트리 모델 정의
#    - 너무 복잡해지지 않게 max_depth/min_samples_leaf로 제약
# =========================================
tree_clf = DecisionTreeClassifier(
    max_depth=4,          # 트리 깊이 제한 (필요하면 3~6 사이에서 조정)
    min_samples_leaf=10,  # 리프 최소 표본 수 (오버핏 방지)
    random_state=42,
)

# =========================================
# 5. 학습
# =========================================
tree_clf.fit(X_train, y_train)

# =========================================
# 6. 평가
# =========================================
y_train_pred = tree_clf.predict(X_train)
y_test_pred  = tree_clf.predict(X_test)

# AUC를 위해 predict_proba 사용
y_train_proba = tree_clf.predict_proba(X_train)[:, 1]
y_test_proba  = tree_clf.predict_proba(X_test)[:, 1]

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
# 7. 피처 중요도
# =========================================
fi = pd.DataFrame({
    "feature": features,
    "importance": tree_clf.feature_importances_,
})
fi = fi.sort_values("importance", ascending=False)
print("\n=== 피처 중요도 (Decision Tree 기준) ===")
print(fi)

# =========================================
# 8. 트리 구조 텍스트로 확인
# =========================================
tree_rules = export_text(tree_clf, feature_names=features)
print("\n=== 디시전 트리 규칙 ===")
print(tree_rules)
