"""실습 06 응용: 04번에서 만든 CV 표만 읽어 선택하고 평가 계획을 기록합니다."""
from pathlib import Path
import pandas as pd
import json
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
# 먼저 04_features.py를 완료하세요. valid_F1을 이용해 후보를 재선택하지 않습니다.
# feature_comparison.csv를 읽어와서 성능이 가장 좋았던 알고리즘을 자동 선택한 후, 
# 이를 최종 평가 계획서(selection.json)에 기록하는 것.

table = pd.read_csv(OUT / "feature_comparison.csv")

# TODO A06: CV_F1이 가장 큰 행의 model 이름을 선택하세요. 정확한 동점은 표의 첫 후보를 사용합니다.
selected = table.loc[table["CV_F1"].idxmax(), "model"]
#ㄴ> 교차검증 F1 점수가 가장 높았던 행의 이름(모델)을 찾아냄. "Logistic" 또는 "Logistic_FE" 중 선택"

manifest = pd.read_csv(HERE / "data/split_manifest.csv")

ids = {name: manifest.loc[manifest["split"] == name].sort_values("position")["request_id"].tolist()
       for name in ["train", "valid", "test"]}

plan = {"selected": selected, "C": 1, "max_iter": 1000, #---------------------------hyperparameter 설정값
        "features": ["input_chars", "allowed_tools", "task_type", "priority"],  #---모델 학습에 사용할 원본 기본 변수 목록
        "derived": selected == "Logistic_FE", "threshold": 0.5, #-------------------파생 변수 조합 선택시 TRUE, 아니면 FALSE
        "rule": "score > threshold", "positive_label": 1,       #-------------------규칙: 확률이 임계값 0.5 초과시 실패로 판정
        "selection": "train60 5-fold mean F1; exact tie: Logistic first", "split_ids": ids}     # 데이터의 실제 ID를 dict 형태로 백업

(OUT / "selection.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
print("선택:", selected)
print("설정: C=1 / 임계값 0.5 초과 / 개발80으로 재학습 / test20 최종 확인")
print("저장: outputs/selection.json (최종평가 이후 변경하지 않습니다.)")
