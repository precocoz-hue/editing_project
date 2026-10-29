"""실습 02: reshape와 broadcasting을 확인합니다."""
import torch

# 내용: 1차원 배열 -> 2차원 행렬로 변환 
#   -> broadcasting(자동확장) 연산 수행 -> GPU로 데이터 이동
a = torch.arange(6, dtype=torch.float32)    #---------1차원 배열 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# 원소 6개를 2행 3열로 바꾸어 matrix에 저장하세요.
# 아래 bias는 각 열에 더해집니다.
# >>> A02
# TODO A02: 위 요구사항을 읽고 이 블록을 완성하세요.
matrix = a.reshape(2, 3)    #-------------------------2차원 배열 [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
# <<< A02
bias = torch.tensor([10., 20., 30.])    #-------------1행 3열짜리 벡터 [10., 20., 30.]

result = matrix + bias  #-----------------------------matrix & bias 행렬 크기 다르지만 pytorch가 자동으로 크기 늘린 후 계산함
# bias가 2행 3열 크기로 복사되어 확장됨
# [[ 0.0,  1.0,  2.0],    +   [[10.0, 20.0, 30.0],
#  [ 3.0,  4.0,  5.0]]         [10.0, 20.0, 30.0]]

print("변환 전:", tuple(a.shape))
print("행렬:", tuple(matrix.shape), matrix.tolist())
print("보정 후:", tuple(result.shape), result.tolist())

# GPU로 데이터 전송
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
moved = result.to(device)

print("이동 장치:", moved.device)
print("CPU에서 확인:", moved.cpu().tolist())    #-------GPU에 있는 상태 그대로 파이썬 리스트로 바꾸면 에러 발생. 안전하게 .cpu()거친 뒤 .tolist()로 출력
"""
<< 결과값 >>
*matrix = a.reshape(2, 3) 일때:
    변환 전: (6,)
    행렬: (2, 3) [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    보정 후: (2, 3) [[10.0, 21.0, 32.0], [13.0, 24.0, 35.0]]
    이동 장치: cuda:0
    CPU에서 확인: [[10.0, 21.0, 32.0], [13.0, 24.0, 35.0]]
    
*"""

