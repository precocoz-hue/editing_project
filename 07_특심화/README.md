# 실습 07 — 회귀·분류 비교 프로젝트 통합 / 심화

PPT 63~73쪽 · 실습 시간 25분. 기본필수를 먼저 완료한 뒤 다음 단계를 선택합니다.

## 목표

응용 결과에 입력 계약·학습 통계 불변 자동 검사를 연결합니다. 새 범주·결측·누락열·K범위 중2가지 이상을 시연합니다.
작업 대상은 **`ex07_compare/03_심화/day6_compare`**입니다. 08에서도 이 폴더를 그대로 사용합니다.


## 진행 순서

먼저 예상 값을 학습기록에 적습니다. 해당 코드의 TODO 함수에 입력·반환값 계약을 확인하고 구현합니다.
한 번에 많은 설정을 바꾸지 말고 현재 단계의 고정 조건을 유지합니다.
실행 후 `expected_output.txt`와 `expected_outputs/`의 수치·shape를 대조하고 자동 검사를 실행합니다.
마지막에 학습기록의 해석 질문에 답합니다. 점수만 기록하고 끝내지 않습니다.

역할 폴더에서 실행합니다.

```bash
python run.py 07 advanced
python tools/check.py --exercise 07 --level advanced
```

직접 실행 파일: `ex07_compare/03_심화/day6_compare/main.py`. 실행 결과: `ex07_compare/03_심화/day6_compare/outputs/`의 비교표·예측·탐색·검사 기록.
01~06을 실행해도 이전 단원 파일을 수정하거나 원본 데이터를 덮어쓰지 않습니다.

## 구현할 함수

### `evaluate.py :: regression_metrics()`

실습02 재사용. MAE_ms, RMSE_ms, R2 dict를 계산한다. 원시 값을 반올림하지 않는다.

### `evaluate.py :: classification_metrics()`

실습03 재사용. labels=[0,1]. Accuracy,Precision,Recall,F1,TN,FP,FN,TP 반환.
실패=1, zero_division=0. 오탐은 FP, 실제 실패를 놓친 것은 FN이다.

### `evaluate.py :: fit_candidates()`

{이름:estimator}별로 독립 Pipeline을 학습한다.
반환 (comparison DataFrame, long prediction DataFrame, fitted_models dict, trace list).
task는 regression/classification. train-only fit, valid predict, 입력·통계 검사를 포함한다.
make_records()는 제공 코드다. 학생은 핵심 학습 흐름을 구현한다.

### `checks.py :: prediction_contract()`

새 입력의 필수 열 검사→transform→predict. 8열/유한값/통계 불변을 확인한다.
반환: n_rows,n_features,statistics_unchanged 딕셔너리. 통계가 바뀌면 AssertionError.
missing/extra 열은 임의로 채우거나 조용히 삭제하지 않고 ValueError로 알린다.

### `search.py :: search_k()`

task,K,scale,weights,p,metric,train_score,valid_score의 10행 표를 만든다.
회귀/분류는 서로 다른 y와 지표를 사용한다. 조건은 standard,uniform,p=2로 고정한다.

### `persist.py :: save_and_verify()`

{task:fitted_pipeline}을 selected_{task}.joblib로 저장하고 즉시 복원·예측 대조한다.
반환 dict의 task별 값: file,sha256,predictions_equal. 자기 실행에서 만든 파일만 load한다.
외부에서 받은 모델 파일은 이 함수에 입력하지 않는다.


## 완료 기준과 제출

README의 목표 수치·반환 형태를 확인하고 `학습기록.md`를 작성합니다.
코드·직접 생성한 결과·검사 로그가 서로 같은 실행에서 나온 것인지 확인합니다.
07~08은 `reports/model_choice.md`와 Git 변경 이력도 제출합니다. test 점수는 제출하지 않습니다.

## 자주 하는 실수

X의 형태/열 순서를 바꾸거나, X와 y를 따로 정렬하거나, valid로 fit하지 마세요.
실수 출력의 마지막 자리 차이는 허용 오차로 비교하고 학습용 예제와 검증용 자료를 구분하세요.
`NotImplementedError`는 TODO 미완성 표시입니다. `expected_outputs`를 코드의 입력으로 읽지 마세요.
