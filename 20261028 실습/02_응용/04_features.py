"""실습 04 응용: 행별 특성 두 개만 추가하고 같은 Logistic·5-fold로 비교합니다."""
from pathlib import Path
import pandas as pd
import numpy as np
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"].tolist()
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"].tolist()
train, valid = raw.loc[train_ids].copy(), raw.loc[valid_ids].copy()
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
NUM, CAT = FEATURES[:2], FEATURES[2:]
X_train, X_valid = train[FEATURES].copy(), valid[FEATURES].copy()
y_train, y_valid = train["failed"], valid["failed"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 제공: 모든 평균·중앙값·범주 목록은 Pipeline.fit의 학습 부분에서만 결정됩니다.
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("scaler", StandardScaler())]), NUM),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.base import clone
# TODO A04: train·valid에 log_input_chars와 chars_per_slot을 추가하세요. 자기 행만 사용하고 NaN은 다음 Pipeline에서 처리합니다.
""" train["log_input_chars"] = np.log1p(train["input_chars"])   # 로그 변환
# 새로운 비율의 feature 생성. 
# 허용된 도구 개수(allowed_tools)대비 입력된 글자 수(input_chars)의 비율이라는 의미를 가진 feature를 생성.
train["chars_per_slot"] = train["input_chars"] / (train["allowed_tools"] + 1)  # 0으로 나누는 것을 방지

valid["log_input_chars"] = np.log1p(valid["input_chars"])   # 로그 변환
# 새로운 비율의 feature 생성. 
# 허용된 도구 개수(allowed_tools)대비 입력된 글자 수(input_chars)의 비율이라는 의미를 가진 feature를 생성.
valid["chars_per_slot"] = valid["input_chars"] / (valid["allowed_tools"] + 1)   # 0으로 나누는 것을 방지

이렇게 작성했었음
"""
for table in (X_train, X_valid):
    table["log_input_chars"] = np.log1p(table["input_chars"])
    table["chars_per_slot"] = table["input_chars"] / (1.0 + table["allowed_tools"])

# TODO 끝
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))

# 목적: 기본 수치형 변수만 썼을 때(Logistic)과 파생 변수 2개 추가할 때(Logistic_FE)중
#      어떤 모델이 더 좋은 성능을 내는지 비교하는 것(교차검증 CV와 검증 valid로 비교분석)
choices = {"Logistic": NUM, "Logistic_FE": NUM + ["log_input_chars", "chars_per_slot"]}
rows = []

for name, numeric_columns in choices.items():
    current_pre = clone(pre)
    # 제공: 파생 열이 있을 때 수치 경로에만 추가합니다. 범주 경로는 바꾸지 않습니다.
    current_pre.transformers[0] = ("num", current_pre.transformers[0][1], numeric_columns)
    # NaN은 Pipeline에서 imputer로 처리됨
    model = Pipeline([("pre", current_pre), ("model", LogisticRegression(C=1, max_iter=1000))])
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1", n_jobs=1)
    model.fit(X_train, y_train)
    rows.append({"model": name, "CV_F1": scores.mean(), "CV_std": scores.std(),
                 "valid_F1": f1_score(y_valid, model.predict(X_valid), zero_division=0)})
result = pd.DataFrame(rows)
result.to_csv(OUT / "feature_comparison.csv", index=False, encoding="utf-8-sig")
print(result.round(4).to_string(index=False))
print("CV는 선택용, valid는 진단용입니다. test는 아직 보지 않습니다.")
"""
<< 결과값 >>
      model  CV_F1  CV_std  valid_F1
   Logistic 0.5219  0.2092    0.3333
Logistic_FE 0.5514  0.1897    0.1818
CV는 선택용, valid는 진단용입니다. test는 아직 보지 않습니다.

"""