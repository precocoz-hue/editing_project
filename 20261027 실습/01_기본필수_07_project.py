"""실습 07~08: 분류 후보 2개를 비교하고 같은 파일을 다시 실행합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 제공: 이 파일 옆의 정상 CSV 20건을 읽습니다. 현재 터미널 위치와 무관합니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")

# 총 20개 데이터set. Train 60%, Valid 20%, Test 20%로 나눠 수행.
# failed 값의 비율은 동일하게 설정
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)

# test 4건은 보관만 합니다. 입력은 실행 전에 아는 글자 수입니다.
X_train = train[["input_chars"]] # 글자수 학습용 60%. 결과물은 2차원 표 형태인 DataFrame으로 반환.
X_valid = valid[["input_chars"]] # 글자수 검증용 20%
y_train = train["failed"]       # failed 학습용 60%. 결과물은 1차원 시리즈로 반환.
y_valid = valid["failed"]       # failed 검증용 20%.
# 관계
# X_train ---> X_valid 문제지
# y_train ---> y_valid 정답지. 이미 DateSet에 있던 정답.
# prediction : X_valid 문제지 풀어서 낸 나름의 정답(=모델의 예측값) 

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import f1_score

# 제공: 전날과 같은 입력 2열입니다. 실행시간과 failed는 입력에 넣지 않습니다.
# valid를 만들기 위해 필요한 features 2개(글자수, 업무종류)
features = ["input_chars", "task_type"]
# X_train: features로 학습
# X_valid: features로 만든 문제지
X_train, X_valid = train[features], valid[features] # 각각 2개의 열이 들어감

# 데이터 가공
pre = ColumnTransformer([
    # num : numerical 숫자 데이터 처리.
    # "input_chars"에 있는 데이터는 숫자형식이기에 passthrough.
    ("num", "passthrough", ["input_chars"]),
    # cat : categorical 범주형/글자 데이터 처리.
    # OneHotEncoder로 "task_type" 데이터를 변환.
    # handle_unknown="ignore": X_valid 검증 데이터(문제지)를 넣었을 때, 컴퓨터가 계산 가능한 0,1 조합으로 변환.
    #                         새로운 글자가 나오면 0으로 채우고 무시할 것.
    # sparse_output=False: 일반적인 표(배열) 형태로 출력
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["task_type"]),
])

print(pre)

# shuffle -> 3개 데이터 fold로 분류(3-fold)
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# DecisionTree: 3개 깊이
# RandomForest: 3개 깊이, 만들고 학습시킬 의사결정나무의 수=20
candidates = [("Tree", DecisionTreeClassifier(max_depth=3, random_state=42)),
              ("Forest", RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=1))]
rows, cv_rows = [], []
predictions = valid[["request_id", "failed"]].copy()

### for문 설명
# 1. Tree & Forest 수행을 위해 pipeline으로 데이터 전처리
# 2. 3-fold 테스트 진행 -> scores 계산
# 3. X_valid 문제지 풀게 만드는 prediction 작성
for name, estimator in candidates:
                    # candidates: #1 수행시 (name="Tree",estimator=DecisionTreeClassifier)
                    #             #2 수행시 (name="Forest",estimator=RandomForestClassifier)

    # 데이터 가공을 통해 생성한 <pre> + 인공지능 모델 <model>을 하나의 set로 묶는 역할
    # Pipeline의 역할: 순서대로 알아서 자동처리.
    #                 "pre"에 데이터 전송 --> OneHotEncoder로 (글자->숫자) 변환.
    #                 "model"(Tree or Forest)에 변환된 데이터 전송.
    # clone(pre)의 역할: Tree 모델 검사할 때 썼던 그대로 사용시 데이터 오염 가능성O.
    #                 매번 깨끗하게 새로 복사된 데이터 사용할 수 있도록 하는 안전장치.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    # TODO B07a: 전처리까지 연결한 model을 train에서 교차검증하세요.
    # X_train 데이터를 3개로 나눠(3-folds) F1-score 계산
    scores = cross_val_score(model,X_train,y_train,cv=folds,scoring='f1',n_jobs=1)
    
    # TODO B07b: 전체 train으로 학습하고 valid를 예측하세요.
    # model.fit(): 전체 학습 데이터를 통째로 줘서 학습시킴. [ 전처리+모델학습 동시에 일어남 ]
    # model.predict(X_valid): 학습된 model에게 X_valid 문제지를 주어 prediction 답안지를 작성하게 만듦
    model.fit(X_train,y_train)
    prediction = model.predict(X_valid)

    # 각 모델의 점수 rows에 추가
    # model: Tree or Forest
    # CV_F1: 3-fold 점수의 평균값
    # CV_std: 3-fold 점수의 표준편차. 값이 작을수록 안정적.
    # valid_F1: 진짜 정답 y_valid와 모델의 답안지(prediction) 비교해서 채점한 최종 결과값
    rows.append({"model": name, "CV_F1": scores.mean(),
                 "CV_std": scores.std(), "valid_F1": f1_score(y_valid, prediction, zero_division=0)})

    # 3-fold 테스트 각각의 결과 세부내역 기록 (위에는 평균점수로 기록됨)
    # enumerate(scores, start=1): (1, 1번 점수), (2, 2번 점수), (3, 3번 점수) 형태로 값 꺼내서 cv_rows에 추가.
    for number, value in enumerate(scores, start=1):
        cv_rows.append({"model": name, "fold": number, "F1": value})

    # 64줄의 predictions에 새로운 열 만들어서 각 모델의 결과값 저장
    # => 표에 predictions["Tree"], predictions["Forest"] 열이 새로 생성됨
    # => [0, 1 ,0 ,0...] 데이터로 차게 되며, 나중에 두 모델이 어떤 문제에서 의견이 갈렸는지 정답을 비교할 때 사용됨     
    predictions[name] = prediction

    # "Forest" 일때만 실행되는 조건문. 어떤 변수를 중요하게 생각했는지 파악하는 분석 표 생성.
    # model["pre"].get_feature_names_out(): OneHotEncoding 되어 늘어난 진짜 변수 이름 불러오기
    # model["model"].feature_importances_: Forest가 각 변수에 준 중요도 점수 불러오기
    if name == "Forest":
        importance = pd.DataFrame({"feature": model["pre"].get_feature_names_out(),
                                   "importance": model["model"].feature_importances_})

# 제공: 입력 파일을 덮어쓰지 않고 결과 4개를 같은 폴더에 만듭니다.
comparison = pd.DataFrame(rows)
comparison.to_csv(HERE / "classification_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(cv_rows).to_csv(HERE / "cv_scores.csv", index=False, encoding="utf-8-sig")
predictions.to_csv(HERE / "validation_predictions.csv", index=False, encoding="utf-8-sig")
importance.to_csv(HERE / "feature_importance.csv", index=False, encoding="utf-8-sig")
print("분할:", len(train), len(valid), len(test))
print(comparison.round(3).to_string(index=False))
print("중요도")
print(importance.round(3).to_string(index=False))
print("저장: classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")

"""
<< 결과값 >>
분할: 12 4 4                    <--- Train 12, Valid 4, Test 4
 model  CV_F1  CV_std  valid_F1
  Tree  0.167   0.236       0.0 
Forest  0.000   0.000       0.0 

# 해석
Tree: CV_F1 0.167 값이지만, 표준편차가 0.236으로 평균보다 큼. 3-fold 집단별로 편차가 큼.
Forest: 전부 0. 데이터가 12개로 적어서 규칙을 찾지 못하고 0으로 결론 내버림.
--------------------------------------------------------------------------
중요도
               feature  importance
      num__input_chars       0.857
     cat__task_type_qa       0.055
cat__task_type_summary       0.088

# 해석
num__input_chars 가 0.857로 모델이 실패를 예측할 때 "input_chars"(글자수)를 1순위로 사용함.
[cat__task_type_qa & cat__task_type_summary] 2개 값은 OneHotEncoder가 task_type을 쪼개 만든 열로, 업무종류에 대한 정보는 예측에 도움이 거의 안 되었다고 판단함.
--------------------------------------------------------------------------
저장: classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv
test는 학습·변환·예측·평가하지 않았습니다.

==========================================================================
<< 결론 >>
1. 모델의 한계
- 데이터 수가 20건으로 너무 적고 실패 사례가 불균형하여 tree, forest 두 경우 모두 실패(failed=1) 패턴 학습 불가. 실제로 F1 점수가 0.0으로 측정됨.

2. 지표 해석
- 실패 여부를 결정짓는 요인은 "input_chars" 글자수 임을 모델 내부 가중치를 통해 알 수 있음

3. 향후 개선 대책
- 모델의 F1 성능을 유의미한 수준으로 끌어올리기 위해 데이터 set에 데이터를 추가로 수집해야 함(최소 수백건).



"""


