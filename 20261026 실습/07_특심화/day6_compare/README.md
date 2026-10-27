# day6_compare — 심화 프로젝트

이 폴더만 복사하여 독립 실행할 수 있습니다. `data/`·환경 목록·보조 모듈을 함께 유지합니다.
실습 07(25분)에서 구현하고 실습 08(35분)에서 같은 프로젝트를 검증합니다. 새로 분할하지 않습니다.

## 실행

이미 준비한 가상환경을 선택한 뒤 실행합니다. Windows에서는 상위 시작 안내의 인터프리터 직접 경로를 사용해도 됩니다.

```bash
python -m pip install -r requirements-lock.txt
python main.py
python verify_outputs.py
python -m unittest -v test_cases
```

수강생용은 TODO 함수를 먼저 구현해야 합니다. `NotImplementedError`는 미완성 함수의 표시입니다.
응용·심화는 `search.py`에서 K 탐색을 재실행하고 `persist.py`에서 선택 Pipeline을 저장합니다. **본인이 직접 생성한 파일임을 확인한 뒤** 다음 복원 검사를 실행합니다.

```bash
python verify_outputs.py --restore --trust-local
```

모델 파일의 해시는 변조 확인용이지 신뢰를 증명하는 수단이 아닙니다. 외부에서 받은 joblib 파일을 이 명령으로 열지 않습니다.

## 파일별 책임

| 파일 | 책임 |
|---|---|
| `main.py` | 두 문제의 후보 학습, 비교, 결과 저장 |
| `preprocessing.py` | 수치형·범주형 전처리 구성 |
| `evaluate.py` | 회귀·분류 지표 계산 |
| `checks.py` | 입력 열·ID·예측 결과·학습 통계 검사 |
| `search.py` | K 후보 기록 생성 또는 체크포인트 재사용 |
| `course_io.py` | 원본과 분할 순서 읽기, 파일 입출력 등 제공 코드 |
| `verify_outputs.py` | 저장한 실제 예측에서 지표를 다시 계산 |
| `test_cases.py` | 정상·경계·실패 입력을 확인하는 제공 테스트 |

## 결과와 제출

`outputs/`의 필수 결과는 `regression_comparison.csv`, `classification_comparison.csv`, `knn_search.csv`, `validation_predictions.csv`입니다.
실행 조건은 `run_metadata.json`, 선택 결과는 `selection.json`, 변환 열은 `feature_names.json`에 남깁니다.
오류 사례 CSV와 `submission_checks.json`도 확인합니다.
보고서는 `reports/model_choice.md`에 작성합니다. 프로그램은 작성한 보고서를 덮어쓰지 않습니다.

현재 폴더의 `Git_기록_안내.md`를 따라 실제 변경 이력을 남깁니다. `.gitignore`로 제외한 결과 파일은 제출용 묶음에 별도로 포함합니다.
상위 단계 폴더의 `expected_outputs/`는 정답 실행의 대조용 기록입니다. 내부 메타데이터의 파일 해시는 참고 실행에 해당하며, 본인의 실행 해시와 같아야 하는 것은 아닙니다.

## 고정 조건

원본·분할 소속표·분할 순서표는 `data/requests.csv`, `split_manifest.csv`, `split_order.csv`입니다.
train 60건에서만 fit하고 valid 20건에서 비교합니다. test 20건은 보관하며 모델 함수에 X·y를 반환하지 않습니다.
`elapsed_ms`와 `failed`는 두 문제의 정답이며 입력 열이 아닙니다.
원본 해시, 실제 입력 열, fit·predict ID, 전처리 통계, 버전을 메타데이터에 기록합니다.

분류의 주 지표는 F1 최대, 회귀는 RMSE 최소입니다. K 실험의 동점은 작은 K를 우선하고, 모델 점수가 완전히 같다면 표에 먼저 등록한 단순 후보를 우선합니다.
이는 실습의 사전 선택 규칙입니다. 작은 valid의 점수 차이로 배포 승인이나 통계적 유의성을 주장하지 않습니다.
기본필수만으로 공통 평가 기준 100점을 충족할 수 있습니다.
