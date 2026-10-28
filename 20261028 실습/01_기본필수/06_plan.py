"""실습 06: test를 보지 않고 train 내부 3-fold로 두 표현을 비교하여 계획을 기록합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")
# 제공: 앞 수업과 같은 seed·층화·행 순서입니다. test 4건은 마지막 평가 전까지 보관합니다.
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)

import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
# 두 파생 연산 모두 자기 행의 글자 수만 사용합니다. 다른 행의 평균·정답을 보지 않습니다.
train = train.copy()
# log_input_chars: 글자 수(input_chars)의 로그 변환
train["log_input_chars"] = np.log1p(train["input_chars"])
# 3-Fold CV를 위해 StratifiedKFold를 사용
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
choices = {"original": ["input_chars"], "with_log": ["input_chars", "log_input_chars"]}
rows = []

# 목적: 글자수 데이터 1개만 주고 3-fold test시 F1 점수와 
#      log_input_chars를 추가했을 때 F1 점수를 비교하여 더 좋은 feature를 선택
# 즉, DecisionTree 모델에게 로그 변환 feature를 만들어준 것이 의미가 있었는지 판단하는 근거가 됨
for name, columns in choices.items():
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    # TODO B06: 같은 folds로 model을 교차검증하세요. train만 넣고 scoring은 f1입니다.
    scores = cross_val_score(
        model, train[columns], train["failed"],     # train 데이터로만 수치형 파생 변수 효과 검증
        cv=folds,
        scoring="f1"
    )
    # TODO 끝
    rows.append({"features": name, "CV_F1": scores.mean(), "CV_std": scores.std()})
comparison = pd.DataFrame(rows)
# 제공: CV 평균 F1 최대를 선택합니다. 정확히 같으면 먼저 적은 original을 고릅니다.
selected = comparison.loc[comparison["CV_F1"].idxmax(), "features"]
plan = {"model": "DecisionTreeClassifier", "max_depth": 3, "random_state": 42,
        "features": choices[selected], "threshold": 0.5, "rule": "score > threshold",
        "positive_label": 1, "selection": "train12 3-fold mean F1; tie: original first",
        "train_ids": train["request_id"].tolist(), "valid_ids": valid["request_id"].tolist(),
        "test_ids": test["request_id"].tolist()}
comparison.to_csv(HERE / "feature_comparison.csv", index=False, encoding="utf-8-sig")
(HERE / "selection.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
print(comparison.round(3).to_string(index=False))
print("선택:", selected, "/ 임계값 0.5 초과일 때 실패")
print("설정 기록: selection.json / test는 아직 예측하지 않았습니다.")
"""
<< 결과값 >>
features  CV_F1  CV_std
original  0.167   0.236
with_log  0.167   0.236
선택: original / 임계값 0.5 초과일 때 실패
설정 기록: selection.json / test는 아직 예측하지 않았습니다.
======================================================
# 해석
DecisionTree 모델에게 log_input_chars를 추가해도 F1 점수가 향상되지 않음.
굳이 복잡하게 변수 늘리지 말고 original만 쓰는 것이 좋음.
원인: log를 씌워도 decisionTree가 만든 분기 규칙에는 영향을 미치지 않기 때문.

"""