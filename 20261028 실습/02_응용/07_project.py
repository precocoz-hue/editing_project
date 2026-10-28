"""실습 07~08 응용: 06번 계획을 읽어 개발80으로 새 Pipeline을 학습하고 test20을 평가합니다."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
plan = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
ids = plan["split_ids"]
develop = raw.loc[ids["train"] + ids["valid"]]
test = raw.loc[ids["test"]]
X_develop, X_test = develop[plan["features"]].copy(), test[plan["features"]].copy()
NUM, CAT = plan["features"][:2], plan["features"][2:]
# 제공: 같은 행별 공식을 새 입력에도 적용합니다. 학습 통계는 아래 fit(develop)에서만 계산됩니다.
# 상단 내용은 이전 단계의 python 파일 내용과 동일함

if plan["derived"]:
    for part in (X_develop, X_test):
        part["log_input_chars"] = np.log1p(part["input_chars"])
        part["chars_per_slot"] = part["input_chars"] / (1.0 + part["allowed_tools"])
    NUM = NUM + ["log_input_chars", "chars_per_slot"]

# 전처리 작업: 결측치 대체, 스케일링 표준화, OneHotEncoding 정의
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())]), NUM),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
])

# 전처리와 모델을 하나의 Pipeline으로 묶음
model = Pipeline([("pre", pre), ("model", LogisticRegression(C=plan["C"], max_iter=plan["max_iter"]))])

# TODO A07a: 새 Pipeline을 X_develop·failed로 학습하세요. 기존 train 통계를 재사용하지 않고 개발80에서 새로 fit합니다.
# 데이터 leakage 방지를 위해 train/valid 통계는 재사용하지 않고, develop에서 새로 fit 하는 것. X_develop 데이터의 평균과 표준편차를 새로 구해서(fit) 스케일링 기준 갱신.
# model.fit 수행시
# 1. 전처리 수행
# 2. LogisticRegression 학습 수행
model.fit(X_develop, develop["failed"])

positive_column = model.classes_.tolist().index(1)

# TODO A07b: 확정 모델로 test의 실패 확률을 구하세요. test는 fit하지 않습니다.
# Test는 fit하지 않는 이유: 새로 들어오는 test data는 모델이 학습한 통계와 다른 분포를 가질 수 있기 때문에, test data를 fit하면 데이터 leakage가 발생할 수 있음.
#                         따라서 test data는 모델이 학습한 통계로만 변환해야 함.
probability = model.predict_proba(X_test)[:, positive_column]

"""
X_develop: train + valid 데이터,
X_test: test 데이터

"""

prediction = (probability > plan["threshold"]).astype(int)
tn, fp, fn, tp = confusion_matrix(test["failed"], prediction, labels=[0, 1]).ravel()
metrics = {"n": len(test), "Accuracy": accuracy_score(test["failed"], prediction),
           "Precision": precision_score(test["failed"], prediction, zero_division=0),
           "Recall": recall_score(test["failed"], prediction, zero_division=0),
           "F1": f1_score(test["failed"], prediction, zero_division=0),
           "ROC_AUC": roc_auc_score(test["failed"], probability),
           "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "actual_failed": int(test["failed"].sum())}
result = pd.DataFrame({"request_id": test.index, "actual": test["failed"].to_numpy(),
                       "probability_failed": probability, "predicted": prediction})
result.to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
(OUT / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print("확정 모델:", plan["selected"], "/ 학습80·최종20")
print(pd.DataFrame([metrics]).round(4).to_string(index=False))
print("오류 사례")
print(result.loc[result["actual"] != result["predicted"]].head(3).round(3).to_string(index=False))
print("저장: outputs/test_predictions.csv, test_metrics.json")
print("08번에서는 이 설정·원본을 그대로 재실행합니다. test를 보고 후보를 다시 고르지 않습니다.")


"""
<< 결과값 >>
확정 모델: Logistic_FE / 학습80·최종20
 n  Accuracy  Precision  Recall     F1  ROC_AUC  TN  FP  FN  TP  actual_failed
20      0.75        0.6     0.5 0.5455   0.7857  12   2   3   3              6
오류 사례
request_id  actual  probability_failed  predicted
      R060       1               0.100          0
      R017       1               0.379          0
      R081       0               0.590          1
저장: outputs/test_predictions.csv, test_metrics.json
08번에서는 이 설정·원본을 그대로 재실행합니다. test를 보고 후보를 다시 고르지 않습니다.
======================================================

# 해석
ROD-AUC=0.786: 0.5 이상으로, 0 or 1 판별 능력이 있는 모델.
Accuracy=0.75: 20건 중 15건을 맞춤.
F1=0.545: 20건 중 6건 실패 중 3건만 맞춤. 데이터 균형이 깨진 상황에서도 절반 정도는 잡아냄.

#오류 사례 분석
- R060: 90% 확률로 성공으로 예측했으나 실제 실패(FN). 예외 사례로, 모델이 학습하지 못한 새로운 유형의 실패가 들어왔을 가능성이 있음.
- R017: 실제 실패인데 확률이 0.5 임계값보다 낮게 나와서 정상으로 잘못 예측(FN)
- R081: 실제 정상인데 확률이 0.5 임계값보다 높게 나와서 실패로 잘못 예측(FP)

# 개선점
- R060, R017: 새로운 유형의 실패를 학습할 수 있도록, 더 많은 학습 자료 확보 필요.
- 실패(1)를 놓쳤을 시 비용이 오탐(FP)보다 크다면, 임계값을 하향조정하여 실패를 추가로 잡아내어 Recall과 F1을 높이는 전략을 고려할 수 있음. (단, Precision은 낮아질 수 있음)

"""