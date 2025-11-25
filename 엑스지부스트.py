import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier

# =========================================
# 0. 데이터 불러오기
# =========================================
file_path = "정제한이벤트완료.xlsx"
df = pd.read_excel(file_path)

print("데이터 크기:", df.shape)

target_col = "is_a_win"
if target_col not in df.columns:
    raise ValueError(f"{target_col} 컬럼이 없습니다.")

# =========================================
# 1. 피처 정의 (main 로지스틱과 동일)
# =========================================
core_features = [
    "duration",
    "A_Reapm",

    "gbr_dropship_landing",
    "gbratktur",
    "gbrinvita",

    "rusfastinv",
    "rusconqnor",

    "fra_invasion",
    "turatkegypt",
    "spaatkusa",

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
# 4. XGBoost 모델 정의 (early stopping 없이, 보수적인 설정)
# =========================================
xgb_clf = XGBClassifier(
    n_estimators=250,        # 너무 크지 않게
    learning_rate=0.05,
    max_depth=3,            # 깊이 줄이기
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

# =========================================
# 5. 학습  ※ early_stopping_rounds, eval_set 없음
# =========================================
xgb_clf.fit(X_train, y_train)

# =========================================
# 6. 평가
# =========================================
y_train_pred = xgb_clf.predict(X_train)
y_test_pred  = xgb_clf.predict(X_test)

y_train_proba = xgb_clf.predict_proba(X_train)[:, 1]
y_test_proba  = xgb_clf.predict_proba(X_test)[:, 1]

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
importances = xgb_clf.feature_importances_
fi = pd.DataFrame({
    "feature": features,
    "importance": importances,
}).sort_values("importance", ascending=False)

print("\n=== XGBoost 피처 중요도 ===")
print(fi)

