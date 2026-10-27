"""실습 07~08 응용: 한 파일에서 회귀·분류 각각 세 후보를 비교합니다."""
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# [1] 데이터 로드 및 학습/검증 데이터 분할 단계
# ---------------------------------------------------------

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# 이 표는 전날의 분할 소속과 각 분할 내부의 행 순서(position)를 함께 보존합니다.
# 학습용 train, 검증용 valid 명단 파일 읽어옴
split = pd.read_csv(HERE / "data/split_manifest.csv")

# split 표에서 split 열의 값 = train인 행 찾고
#                       -> 내부 순서 "position" 대로 정렬
#                       -> ID("request_id")만 뽑아오기
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]

# raw.loc[ID명단] = 위에서 구한 ID명단으로 원본 전체 데이터(raw)에서 학습용/검증용 데이터 각각 분리
train = raw.loc[train_ids]  # 학습용 전체 데이터
valid = raw.loc[valid_ids]  # 검증용 전체 데이터

# 모델 학습에 사용할 4개의 열 목록을 리스트 형식으로 지정
features = ["input_chars", "allowed_tools", "task_type", "priority"]

# 전체 데이터에서 모델에게 줄 X 값만 따로 추출
X_train = train[features]
X_valid = valid[features]

# ---------------------------------------------------------
# [2] 전처리(Preprocessing) 설계 단계
# ---------------------------------------------------------

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix

# 제공: 전날 만든 전처리입니다. 숫자 2열은 중앙값→표준화, 문자 2열은 최빈값→원핫입니다.
# Pipeline([...]): 데이터 변환 단계를 하나로 묶어주는 파이프라인
# SimpleInputer(strategy="median"): 빈 칸(결측치)이 있으면 해당 열의 중앙값으로 채움
# StandardScaler(): 데이터 평균을 0, 표준편차 1로 맞추는 표준화 작업
num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

# SimpleInputer(strategy="most_frequent"): 빈 칸(결측치)O -> 최빈값으로 채움
# OneHotEncoder(...): 글자 데이터-> 2진수로 변환
cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

# ColumnTransformer([...]): 데이터의 열(column)별로 서로 다른 전처리 적용
# features[:2]->앞의 2개 열("input_chars", "allowed_tools")는 숫자형. num 전처리 적용.
# features[2:]->뒤의 2개 열("task_type", "priority")는 문자/범주형. cat 전처리 적용.
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

# ---------------------------------------------------------
# [3] 회귀(Regression) 모델 루프: 연속적인 숫자(작업 시간 등) 예측
# ---------------------------------------------------------

# 제공 K: PPT 실습06의 동일 원본·분할·standard 탐색 결과(분류3, 회귀15)를 재사용합니다.
regression_rows = []        # 각 회귀 모델의 성적표 점수를 담을 리스트
classification_rows = []    # 
predictions = []            # 실제값과 모델들의 예측값을 한데 모아 비교할 표 데이터를 담을 리스트

# 3가지 서로 다른 알고리즘 후보를 순서대로 돌림
# - DummyRegressor(): 아무런 학습 없이 평균값으로 찍는 기본 베이스라인 모델
# - LinearRegression: 수학 선형 공식 사용. 전통적인 선형 회귀 모델.
# - KNeighborsRegressor: KNN 회귀. 주변 비슷한 데이터 평균값으로 맞추고, n=15로 설정됨.
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression()),
                        ("KNN15", KNeighborsRegressor(n_neighbors=15))]:

    # clone(pre): 전처리 틀 pre 복사
    # Pipeline(...): 전처리 pre와 머신러닝 알고리즘(estimator)을 하나로 결합
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    # TODO A07r: 회귀 모델을 학습하고 valid 예측을 구하세요. 힌트: 정답 열은 elapsed_ms입니다.
    # model.fit(문제, 정답): 모델 학습시킴. 정답은 "elapsed_ms"를 사용해서 얻음.
    model.fit(X_train,train["elapsed_ms"])
    # model.predict(문제): 학습된 모델에게 새로운 문제지를 주고 예측값을 받음.
    prediction = model.predict(X_valid) # X_valid는 30줄에서 정의됨

    # 성적표 리스트에 모델 이름과 오차 점수를 계산해서 저장
    # mean_absolute_error: 실제값과 예측값의 절대적인 차이(평균 오차값)
    # root_mean_squred_error: RMSE. 큰 오차에 벌점을 더 많이 주는 방식의 평균 오차.
    regression_rows.append({"model": name, "scale": "standard",
        "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
        "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction)})

    # 추후 파일 저장을 위해 [요청ID, 실제정답, 예측값]을 표(DataFrame)로 묶어 리스트에 기록
    predictions.append(pd.DataFrame({"request_id": valid.index, "task": "regression", "model": name,
        "actual": valid["elapsed_ms"].to_numpy(), "predicted": prediction}))

# ---------------------------------------------------------
# [4] 분류(Classification) 모델 루프: 이진 상태(성공 0 / 실패 1) 예측
# ---------------------------------------------------------
# classification_rows = [] : 각 분류 모델의 성적표 점수를 담을 리스트. 69번 줄에 있음.

# - DummyClassifier: 학습 없이 가장 빈도가 높은 정답으로 찍는 베이스라인 모델
# - LogisticRegression: 0과 1 사이의 확률을 계산해 성공/실패를 이진 분류하는 선형 모델
# - KNeighborsClassifier: 주변에 가장 비슷한 3개 데이터가 다수결로 결정하는 KNN 분류 모델
for name, estimator in [("Dummy", DummyClassifier(strategy="most_frequent")),
                        ("Logistic", LogisticRegression(C=1.0, max_iter=1000)),
                        ("KNN3", KNeighborsClassifier(n_neighbors=3))]:
    # 작업 과정은 회귀와 동일
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    # TODO A07c: 분류 모델을 학습하고 valid 예측을 구하세요. 힌트: 정답 열은 failed입니다.
    # model.fit(문제, 정답): "failed"인 경우를 기준으로 학습시킴.
    model.fit(X_train,train["failed"])
    # model.prediction(문제): 학습된 분류 모델에게 문제를 주고 0 or 1의 예측 결과를 받아옴
    prediction = model.predict(X_valid)

    # 혼동행렬값 구하기
    # .ravel(): 2차원 행렬 표를 계산하기 편하게 1차원 줄(리스트)로 한 줄로 쭉 펼쳐줍니다.
    tn, fp, fn, tp = confusion_matrix(valid["failed"], prediction, labels=[0, 1]).ravel()
    
    # 분류 성적표 리스트에 모델의 이름과 다양한 평가지표 점수들 계산 후 저장
    classification_rows.append({"model": name, "scale": "standard",
        "F1": f1_score(valid["failed"], prediction, zero_division=0),
        "Precision": precision_score(valid["failed"], prediction, zero_division=0),
        "Recall": recall_score(valid["failed"], prediction, zero_division=0), "FP": fp, "FN": fn})
    predictions.append(pd.DataFrame({"request_id": valid.index, "task": "classification", "model": name,
        "actual": valid["failed"].to_numpy(), "predicted": prediction}))

# ---------------------------------------------------------
# [5] 결과 취합 및 파일 저장, 콘솔 출력 단계
# ---------------------------------------------------------

# 회귀 결과표
regression = pd.DataFrame(regression_rows)
# 분류 결과표
classification = pd.DataFrame(classification_rows)
out = HERE / "outputs"
out.mkdir(exist_ok=True)

# 화면에 결과 출력
regression.to_csv(out / "regression_comparison.csv", index=False, encoding="utf-8-sig")
classification.to_csv(out / "classification_comparison.csv", index=False, encoding="utf-8-sig")
pd.concat(predictions, ignore_index=True).to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8-sig")
print("회귀 / valid 20건")
print(regression.round(3).to_string(index=False))
print("분류 / valid 20건")
print(classification.round(3).to_string(index=False))
print("저장: outputs의 비교표 2개와 validation_predictions.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")


"""
<< 결과값 >>
회귀 / valid 20건
 model    scale  MAE_ms  RMSE_ms
 Dummy standard 541.260  658.745
Linear standard 551.797  639.996
 KNN15 standard 530.063  641.788
분류 / valid 20건
   model    scale    F1  Precision  Recall  FP  FN
   Dummy standard 0.000      0.000   0.000   0   6
Logistic standard 0.333      0.333   0.333   4   4
    KNN3 standard 0.545      0.600   0.500   2   3
저장: outputs의 비교표 2개와 validation_predictions.csv
test는 학습·변환·예측·평가하지 않았습니다.

"""