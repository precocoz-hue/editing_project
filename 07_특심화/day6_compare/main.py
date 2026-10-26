"""Day6 회귀·분류 모델 비교. 프로젝트 폴더를 단독 복사해도 실행됩니다."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.dummy import DummyClassifier,DummyRegressor
from course_io import load_data,read_table,write_json,environment,FEATURES,sha256
from evaluate import fit_candidates,select_models
from search import choose_k
from checks import prediction_contract

ROOT=Path(__file__).resolve().parent
LEVEL='advanced'

def run_project(data_dir=None,output_dir=None):
    data_dir=Path(data_dir) if data_dir else ROOT/'data'
    out=Path(output_dir) if output_dir else ROOT/'outputs'
    d=load_data(data_dir)
    if LEVEL=='basic':
        # 실습06 결과 재사용. 제공 파일을 계산 결과인 것처럼 숨기지 않고 출처를 기록합니다.
        own=data_dir/'knn_search_from_ex06.csv'
        ref=own if own.exists() else data_dir/'knn_search_reference.csv'
        search=pd.read_csv(ref)
        source='student_ex06_record' if own.exists() else 'provided_ex06_checkpoint'
    else:
        from search import search_k
        search=search_k(d)
        source='recomputed_train_valid'
    ks=choose_k(search)
    reg_spec={'Dummy':DummyRegressor(strategy='mean'), 'Linear':LinearRegression(),
              f"KNN_k{ks['regression']}":KNeighborsRegressor(n_neighbors=ks['regression'])}
    cls_spec={'Dummy':DummyClassifier(strategy='most_frequent'),
              'Logistic':LogisticRegression(C=1.0,max_iter=1000),
              f"KNN_k{ks['classification']}":KNeighborsClassifier(n_neighbors=ks['classification'])}
    rt,rp,rm,rtrace=fit_candidates(reg_spec,d.X_train,d.yr_train,d.X_valid,d.yr_valid,'regression')
    ct,cp,cm,ctrace=fit_candidates(cls_spec,d.X_train,d.yc_train,d.X_valid,d.yc_valid,'classification')
    selection=select_models(rt,ct)

    # 두 문제에서 쓰지 않는 보조 열도 float로 맞춰 all-NA concat 경고를 방지합니다.
    for table in [rp,cp]:
        table[['residual_ms','probability_failed']]=table[['residual_ms','probability_failed']].astype(float)
    predictions=pd.concat([rp,cp],ignore_index=True)

    # test를 모델에 전달한 흔적은 없어야 합니다. ID/개수만 정책 기록에 포함합니다.
    if set(predictions.request_id)&set(d.test_ids): raise AssertionError('test ID가 예측에 섞였습니다.')
    chosen={'regression':rm[selection['regression']['model']],
            'classification':cm[selection['classification']['model']]}
    new=read_table(data_dir/'new_requests.csv').set_index('request_id')
    contracts={task:prediction_contract(m,new) for task,m in chosen.items()}
    out.mkdir(parents=True,exist_ok=True)

    for name,table in [('regression_comparison',rt),('classification_comparison',ct),
                        ('knn_search',search),('validation_predictions',predictions)]:
        table.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')

    # 오류 사례를 보고서에 옮기기 쉽도록 ID·실제·예측을 함께 보존합니다.
    errors=rp.loc[rp.model==selection['regression']['model']].copy()
    errors['abs_error_ms']=errors.residual_ms.abs()
    errors.sort_values('abs_error_ms',ascending=False,kind='stable').head(3).to_csv(
        out/'largest_regression_errors.csv',index=False,encoding='utf-8-sig')
    
    class_errors=cp.loc[(cp.model==selection['classification']['model'])&(cp.y_true!=cp.y_pred)].copy()
    class_errors['error_type']=np.where(class_errors.y_true==1,'FN','FP')
    class_errors.to_csv(out/'classification_errors.csv',index=False,encoding='utf-8-sig')
    write_json(out/'selection.json',selection)
    write_json(out/'feature_names.json',chosen['classification']['pre'].get_feature_names_out().tolist())
    
    metadata={'level':LEVEL,'environment':environment(),'source_hashes':d.source_hashes,
        'n_train':len(d.X_train),'n_valid':len(d.X_valid),'n_test_reserved':len(d.test_ids),
        'train_ids':d.X_train.index.tolist(),'valid_ids':d.X_valid.index.tolist(),
        'test_ids_reserved':d.test_ids,'input_columns':FEATURES,'test_used_for_model':False,
        'test_policy':'ID 무결성 확인만, 변환·예측·평가·선택에는 미사용',
        'search_source':source,'search_conditions':{'scale':'standard','K':[1,3,5,9,15],'weights':'uniform','p':2},
        'selection_policy':{'regression':'valid RMSE 최소','classification':'valid F1 최대',
                            'K_tie':'작은 K','model_tie':'먼저 등록한 단순 후보'},
        'fit_trace':rtrace+ctrace,'new_input_contract':contracts,
        'persistence':'not_requested_at_basic'}
    
    if LEVEL!='basic':
        from persist import save_and_verify
        metadata['persistence']=save_and_verify(chosen,d.X_valid,out)

    write_json(out/'run_metadata.json',metadata)
    print('train / valid / reserved test:',len(d.X_train),len(d.X_valid),len(d.test_ids))
    print('comparison rows:',len(rt),len(ct),'prediction rows:',len(predictions),'search rows:',len(search))
    print('classification:',selection['classification']['model'],f"F1={selection['classification']['value']:.3f}")
    print('regression:',selection['regression']['model'],f"RMSE={selection['regression']['value']:.2f} ms")
    print('test_used_for_model:',False)
    return {'regression':rt,'classification':ct,'predictions':predictions,'search':search,
            'selection':selection,'metadata':metadata}

def main():
    # argparse.ArgumentParser(): 사용자가 --옵션 형태로 입력하는 명령어 인자를
    #                            쉽게 잘라내어 파이썬 변수로 만들어주는 내장 라이브러리 함수.
    # (description=__doc__): 이 파일 맨 위에 적힌 설명 주석(__doc__)을 그대로 가져다 쓰겠다는 뜻
    parser=argparse.ArgumentParser(description=__doc__)
    # '--data-dir': 데이터 폴더 경로 등록
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    # '--output-dir': 데이터 결과 저장 폴더 경로 등록
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    # parse_args(): 명령어 해석(파싱) 후 args 변수에 저장. args.data_dir 형태로 꺼내쓸 수 있게 변환됨.
    args=parser.parse_args()
    # 컴퓨터 자원 이용 관련 함수. 쓰레드 사용 개수 제한(다운 방지)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1): run_project(args.data_dir,args.output_dir)

if __name__=='__main__': main()
