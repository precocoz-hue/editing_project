"""실습06의 K 실험을 종합 프로젝트로 옮깁니다. test는 함수 인자로도 받지 않습니다."""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from preprocessing import build_pipeline
from evaluate import regression_metrics,classification_metrics
from course_io import validate_k

def search_k(d,candidates=(1,3,5,9,15)):
    """task,K,scale,weights,p,metric,train_score,valid_score의 10행 표를 만든다.
    회귀/분류는 서로 다른 y와 지표를 사용한다. 조건은 standard,uniform,p=2로 고정한다.
    """
    # TODO D6-07-H4: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.
    raise NotImplementedError("TODO D6-07-H4: 구현 후 다시 실행하세요.")

def choose_k(table):
    result={}
    for task in ['classification','regression']:
        p=table.loc[table.task==task].sort_values(['valid_score','K'],
                    ascending=[task=='regression',True],kind='stable')
        if p.empty: raise ValueError('K 탐색표에 두 문제의 결과가 모두 필요합니다.')
        result[task]=int(p.iloc[0]['K'])
    return result
