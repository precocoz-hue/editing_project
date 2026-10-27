"""지표·학습·검증 담당. 비교표의 점수는 모두 validation 결과입니다."""
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error,root_mean_squared_error,r2_score,
    accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)
from preprocessing import build_pipeline
from checks import before_model,check_finite
from course_io import snapshot

# y: 실제 정답 / pred: 모델의 예측값
def regression_metrics(y,pred):
    """실습02 재사용. MAE_ms, RMSE_ms, R2 dict를 계산한다. 원시 값을 반올림하지 않는다."""
    # TODO D6-07-H1: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.
    # sklearn에서 함수를 불러와서 작성 가능.
    metrics = {
        "MAE_ms": mean_absolute_error(y,pred),
        "RMSE_ms": root_mean_squared_error(y,pred),
        "R2": r2_score(y,pred)
    }
    return metrics

def classification_metrics(y,pred):
    """실습03 재사용. labels=[0,1].
     Accuracy,Precision,Recall,F1,TN,FP,FN,TP 반환.
    실패=1, zero_division=0. 오탐은 FP, 실제 실패를 놓친 것은 FN이다.
    """
    # TODO D6-07-H2: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.

    # 혼동행렬
    # .ravel(): 2차원 행렬 표를 계산하기 편하게 1차원 줄(리스트)로 한 줄로 쭉 펼쳐줍니다.
    tn, fp, fn, tp = confusion_matrix(y,pred, labels=[0, 1]).ravel()

    # 조건에 맞게 평가지표 딕셔너리 설정
    # 조건: 실패=1, zero_division=0. 오탐은 FP, 실제 실패를 놓친 것은 FN.
    # zero_division=0 : 정답, 예측값 중 특정 클래스가 아예 없으면 분모가 0이 됨.
    #                   에러 대신 0이 나오도록 하는 안전장치.
    metrics = {
        "Accuracy":accuracy_score(y,pred),
        "Precision":precision_score(y,pred,zero_division=0),
        "Recall":recall_score(y,pred,zero_division=0),
        "F1":f1_score(y,pred,zero_division=0),
        "TN":int(tn),
        "FP":int(fp),
        "FN":int(fn),
        "TP":int(tp)
    }
    return metrics

# 머신러닝 알고리즘 후보를 한 번에 훈련시킨 후 성능을 측정하여 비교표를 뽑는 과정
def fit_candidates(candidates,X_train,y_train,X_valid,y_valid,task):
    """{이름:estimator}별로 독립 Pipeline을 학습한다.
    반환 (comparison DataFrame, long prediction DataFrame, 
    fitted_models dict, trace list).
    task는 regression/classification. train-only fit, valid predict, 
    입력·통계 검사를 포함한다.
    make_records()는 제공 코드다. 학생은 핵심 학습 흐름을 구현한다.
    """
    # TODO D6-07-H3: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.
    # candidates: 후보군 딕셔너리
    # X_train, y_train: 학습용 데이터
    # X_valid, y_valid: 검증용 데이터
    # task: 작업 종류(회귀,분류 등)
    
    # 결과들을 담아둘 빈 그릇들
    rows = []           # 성능지표(comparision DataFrame용)
    predictions = []    # 실제값 vs 예측값 표
    fitted_models = {}  # 학습 완료된 모델 저장용 dict
    trace = []          # 입력/통계 검사 기록용 list

    # 제공된 전처리기(pre)가 상위 스크립트에 있다고 가정
    for name, estimator in candidates.items():
        # 1. 파이프라인 생성 (전처리 clone 필수)
        model = Pipeline([("pre",clone(pre)),("model",estimator)])

        # 2. 학습 (train-only fit) 
        # fit: 훈련용 데이터로만 학습
        model.fit(X_train, y_train)
        fitted_models[name] = model     # 학습된 모델 보관

        # 3. 예측 (valid predict)
        # predict: 검증용 데이터로 예측
        pred = model.predict(X_valid)

        # 4. 점수 계산 (task에 따라 다르게 적용)
        if task == "classification":
            metrics = classification_metrics(y_valid,pred)
            metrics["model"] = name     # 어떤 모델의 점수인지 기록
            classification_row.append(metrics)

            # 예측값 기록용 데이터 프레임 생성
            pred_df = pd.DataFrame({
                "request_id":X_valid.index, "task":"classification", "model":name,
                "actual":y_valid.to.numpy(), "predicted":pred
            })
            predictions.append(pred_df)

        elif task == "regression":
            metrics = regression_metrics(y_valid,pred)
            metrics["model"] = name
            regression_row.append(metrics)

            pred_df = pd.DataFrame({
                "request_id":X_valid.index, "task":"regression", "model":name,
                "actual":y_valid.to.numpy(), "predicted":pred
            })
            predictions.append(pred_df)

        if task == "classification":
            comparision_df = pd.DataFrame(classification_rows)
        else:
            comparision_df = pd.DataFrame(regression__rows)

        prediction_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()

        return comparision_df, prediction_df, fitted_models, trace
"""
여기까지 코드 답안과 비교하며 작성
"""



def make_records(task,name,ids,y,pred,model,X):
    """행 순서를 유지하는 제공 코드. 회귀 잔차는 실제-예측입니다."""
    probability=None
    if task=='classification':
        positive=list(model['model'].classes_).index(1)
        probability=model.predict_proba(X)[:,positive]
    records=[]
    for i,request_id in enumerate(ids):
        row={'request_id':request_id,'split':'valid','task':task,'model':name,
             'y_true':float(y.iloc[i]),'y_pred':float(pred[i]),
             'residual_ms':float(y.iloc[i]-pred[i]) if task=='regression' else None,
             'probability_failed':float(probability[i]) if probability is not None else None}
        records.append(row)
    return records

def select_models(regression,classification):
    """제공: 사전 지표로 정렬. 완전 동점이면 먼저 등록한 단순 후보를 우선합니다."""
    r=regression.sort_values('RMSE_ms',ascending=True,kind='stable').iloc[0]
    c=classification.sort_values('F1',ascending=False,kind='stable').iloc[0]
    return {'regression':{'model':r['model'],'metric':'RMSE_ms','value':float(r['RMSE_ms'])},
            'classification':{'model':c['model'],'metric':'F1','value':float(c['F1'])}}
