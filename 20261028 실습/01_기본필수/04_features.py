"""실습 04: 다른 행의 통계나 정답을 사용하지 않는 파생 열 하나를 만듭니다."""
from pathlib import Path
import pandas as pd
import numpy as np
raw = pd.read_csv(Path(__file__).resolve().parent / "requests.csv")
# 원본 파일은 덮어쓰지 않습니다. 글자 수만 복사하고 정답·실행시간은 선택하지 않습니다.
features = raw[["input_chars"]].copy()

# TODO B04: np.log1p로 글자 수의 로그 변환 열을 추가하세요. 각 행의 값만 사용합니다.
# 데이터의 특정 특징(feature)를 가공하는 것
# 목적: 글자수(input_chars) 데이터가 가진 불균형한 분포를 로그 변환을 통해 다듬는 작업
features["log_input_chars"] = np.log1p(features["input_chars"]) # 행별로 연산 적용
# TODO 끝

print(features.head().round(3).to_string(index=False))
print("행/열:", features.shape)
print("원본 열:", raw.columns.tolist())
"""
<< 결과값 >>
 input_chars  log_input_chars
         100            4.615   <--- 로그 취하니 숫자 분포가 줄어듦
         200            5.303   <--- 상동
         300            5.707   <--- 상동
         400            5.994   <--- 상동
         500            6.217   <--- 상동
행/열: (20, 2)
원본 열: ['request_id', 'input_chars', 'task_type', 'failed', 'elapsed_ms']

"""