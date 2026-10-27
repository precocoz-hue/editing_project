"""Day5·실습05 재사용 코드. 새 전처리를 추가하지 않고 모델만 교체합니다."""
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
NUM=['input_chars','allowed_tools']
CAT=['task_type','priority']

def make_preprocessor():
    numeric=Pipeline([('imputer',SimpleImputer(strategy='median')),
                      ('scaler',StandardScaler())])
    categorical=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    return ColumnTransformer([('num',numeric,NUM),('cat',categorical,CAT)],remainder='drop')

def build_pipeline(estimator):
    # 한 후보의 fit이 다른 후보의 fitted 상태를 덮어쓰지 않도록 매번 새로 만듭니다.
    return Pipeline([('pre',make_preprocessor()),('model',clone(estimator))])
