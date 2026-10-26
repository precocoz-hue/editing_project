"""실습08 제공 검사 도구. 기존 결과를 읽어서 검증하며 새로 fit하지 않습니다."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from course_io import load_data,write_json,reg_metrics,cls_metrics,sha256,environment
ROOT=Path(__file__).resolve().parent

def verify_output(output_dir=None,data_dir=None):
    out=Path(output_dir) if output_dir else ROOT/'outputs'
    # 재검사 시작 시 이전 성공 기록을 지워 오래된 결과를 오인하지 않게 합니다.
    (out/'submission_checks.json').unlink(missing_ok=True)
    d=load_data(Path(data_dir) if data_dir else ROOT/'data')
    paths=['regression_comparison.csv','classification_comparison.csv','knn_search.csv',
           'validation_predictions.csv','run_metadata.json','selection.json','feature_names.json']
    missing=[p for p in paths if not (out/p).is_file()]
    if missing: raise FileNotFoundError('먼저 main.py 실행 필요: '+', '.join(missing))
    reg=pd.read_csv(out/paths[0]); cls=pd.read_csv(out/paths[1]); search=pd.read_csv(out/paths[2])
    preds=pd.read_csv(out/paths[3]); meta=json.loads((out/'run_metadata.json').read_text(encoding='utf-8'))
    results=[]
    def check(name,condition):
        if not condition: raise AssertionError(name)
        results.append({'check':name,'passed':True})
    check('비교표는 문제별 3행',len(reg)==3 and len(cls)==3)
    check('원본·분할표·순서표가 실행 시점과 동일',meta['source_hashes']==d.source_hashes)
    check('test 정책',meta['test_used_for_model'] is False)
    check('valid 예측은 120행',len(preds)==len(d.X_valid)*6)
    check('예측 key는 고유',not preds.duplicated(['task','model','request_id']).any())
    check('test ID 없음',not set(preds.request_id)&set(d.test_ids))
    check('test 점수 열 없음',not any(c.startswith('test_') for t in [reg,cls,search,preds] for c in t.columns))
    for task,table,target,metric_fn in [('regression',reg,d.yr_valid,reg_metrics),
                                      ('classification',cls,d.yc_valid,cls_metrics)]:
        for row in table.to_dict('records'):
            p=preds.loc[(preds.task==task)&(preds.model==row['model'])]
            check(f'{task}/{row["model"]}: ID 순서',p.request_id.tolist()==d.X_valid.index.tolist())
            check(f'{task}/{row["model"]}: 실제값 정렬',np.array_equal(p.y_true.to_numpy(),target.to_numpy()))
            vals=metric_fn(target,p.y_pred.to_numpy())
            check(f'{task}/{row["model"]}: 지표 재계산',all(np.isclose(row[k],v,rtol=1e-9,atol=1e-9) for k,v in vals.items()))
            check(f'{task}/{row["model"]}: 예측 유한',np.isfinite(p.y_pred).all())
    check('전처리 학습은 train만',all(r['fit_ids']==d.X_train.index.tolist() and r['statistics_unchanged'] for r in meta['fit_trace']))
    check('전처리 8열',len(json.loads((out/'feature_names.json').read_text(encoding='utf-8')))==8)
    check('K 탐색은 두 문제·5후보',len(search)==10 and set(search.K)=={1,3,5,9,15})
    selection=json.loads((out/'selection.json').read_text(encoding='utf-8'))
    check('지표 방향에 맞는 선택',
          selection['regression']['model']==reg.sort_values('RMSE_ms',kind='stable').iloc[0]['model'] and
          selection['classification']['model']==cls.sort_values('F1',ascending=False,kind='stable').iloc[0]['model'])
    check('허용 입력 4열',meta['input_columns']==['input_chars','allowed_tools','task_type','priority'])
    write_json(out/'submission_checks.json',results)
    print(f'결과 파일 검사 {len(results)}건 통과. 보고서·개인 설명·Git 이력은 별도 확인합니다.')
    return results

def verify_saved(output_dir=None,trust_local=False):
    """자체 생성 파일임을 확인한 경우만 호출. hash는 변조 탐지이지 신뢰의 증명이 아닙니다."""
    if not trust_local: raise ValueError('직접 생성한 모델인지 확인한 뒤 --trust-local을 지정하세요.')
    import joblib
    out=Path(output_dir) if output_dir else ROOT/'outputs'
    meta=json.loads((out/'run_metadata.json').read_text(encoding='utf-8'))
    if not isinstance(meta['persistence'],dict): raise ValueError('기본필수 단계에는 저장 모델이 없습니다.')
    if meta['environment']!=environment():
        raise RuntimeError('학습·복원 라이브러리/Python 버전이 다릅니다. 같은 환경에서 재생성하세요.')
    d=load_data(ROOT/'data'); p=pd.read_csv(out/'validation_predictions.csv')
    selection=json.loads((out/'selection.json').read_text(encoding='utf-8'))
    result={}
    for task,record in meta['persistence'].items():
        filename=f'selected_{task}.joblib'
        if record['file']!=filename: raise ValueError('허용되지 않은 모델 파일명')
        path=out/filename
        if sha256(path)!=record['sha256']: raise ValueError('저장 모델 해시 불일치')
        model=joblib.load(path)  # 명시적 신뢰 확인 + 해당 실행 기록의 파일만 로드
        expected=p.loc[(p.task==task)&(p.model==selection[task]['model']),'y_pred'].to_numpy()
        equal=bool(np.allclose(model.predict(d.X_valid),expected,rtol=1e-10,atol=1e-10))
        if not equal: raise AssertionError('새 프로세스 복원 예측이 원본 결과와 다름')
        result[task]=equal
    write_json(out/'restore_checks.json',result)
    print('새 프로세스 복원 대조:',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    p.add_argument('--trust-local',action='store_true',help='이 컴퓨터에서 직접 생성한 모델만 복원')
    p.add_argument('--restore',action='store_true')
    a=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        verify_output(a.output_dir)
        if a.restore: verify_saved(a.output_dir,a.trust_local)

if __name__=='__main__': main()
