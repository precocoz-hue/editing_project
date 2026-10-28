"""실습 02 응용: 가상 확률로 임계값만 바꿉니다. 최종 모델의 임계값을 탐색하는 코드가 아닙니다."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
y = np.array([0, 0, 1, 1])
score = np.array([0.10, 0.40, 0.35, 0.80])
rows = []

# threshold(임곗값) 0.5 일때와 0.3 일때 비교하는 시뮬레이션
for threshold in [0.5, 0.3]:
    # TODO A02: 확률이 threshold(임곗값)보다 클 때만 실패1로 바꾸세요. 같은 값은 성공0입니다.
    prediction = (score > threshold).astype(int)        #ppt p24 참고
    # TODO 끝

    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    # 정상을 실패로 오해한 비용(FP)보다 실패를 못잡고 놓친 비용(FN)의 대가가 5배 크다고 가정함.
    rows.append({"threshold": threshold, "FP": fp, "FN": fn, "F1": f1_score(y, prediction),
                 "assumed_cost": fp + 5*fn, "AUC": roc_auc_score(y, score)})
result = pd.DataFrame(rows)
print(result.round(3).to_string(index=False))
result.to_csv(OUT / "threshold_comparison.csv", index=False, encoding="utf-8-sig")
print("cost는 FP 비용1, FN 비용5라는 가정입니다. 최종 평가 임계값은 사전에 0.5로 고정합니다.")
"""
<< 결과값 >>
 threshold  FP  FN    F1  assumed_cost  AUC
       0.5   0   1 0.667             5 0.75
       0.3   1   0 0.800             1 0.75
cost는 FP 비용1, FN 비용5라는 가정입니다. 최종 평가 임계값은 사전에 0.5로 고정합니다.

# 해석
threshold = 0.5 일때:
- 실제 실패 1건 놓침
 => F1-score = 0.667, assumed_cost = 5

threshold = 0.3 일때:
- 기준을 낮춰 조금이라도 의심되면 일단 실패로 경고하는 전략
- 정상 1건을 불량으로 오해하는 실수 발생(FP=1), 실패는 다 잡아냄(FN=0)
 => F1-score = 0.800, assumed_cost = 1 

## 결론
임계값을 0.3으로 낮추는 것이 유리함

"""