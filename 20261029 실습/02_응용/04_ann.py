"""실습 04: 은닉 너비에 따라 파라미터 수가 변합니다. 학습 성능 비교는 아닙니다."""
import torch
from torch import nn

# 내용: pytorch를 사용해 신경망 기초뼈대 만든 후 은닉층(Hidden Layer)의 크기에 따라 
#      모델의 크기(parameter 수)가 어떻게 변하는지 확인하는 코드
torch.manual_seed(42)
X = torch.ones(4, 2)
for hidden in [4, 8, 16]:
    # hidden 값을 사용해 입력 2개 → 은닉 hidden개 → 출력 2개를 연결하세요.
    # 은닉층 뒤에 ReLU를 넣고 마지막에 softmax는 넣지 않습니다.
    # >>> A04
    # TODO A04: 위 요구사항을 읽고 이 블록을 완성하세요.
    # 입력층) nn.Linear(2, hidden): 처음에 2개 데이터 입력 받고 hidden개로 들려서 내보냄
    # 활성화) nn.ReLU(): 늘어난 데이터에 비선형성(복잡함)을 더해줌 + 앞뒤 linear가 병합되는 것을 방지함(값의 다양성이 떨어짐)
    # 출력층) nn.Linear(hidden, 2): hidden개의 데이터를 최종 결과물 2개로 압축하여 출력
    model = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 2))
    # <<< A04

    logits = model(X)
    count = sum(p.numel() for p in model.parameters())
    print("은닉:", hidden, "출력:", tuple(logits.shape), "파라미터:", count)
"""
<< 결과값 >>
은닉: 4 출력: (4, 2) 파라미터: 22 
[레이어1] 가중치 2*4+ 편향bias 4, 총 12 
[레이어2] 가중치 2*4+ 편향bias 2, 총 10

은닉: 8 출력: (4, 2) 파라미터: 42
[레이어1] 가중치 2*8 +bias 8, 총 24
[레이어2] 가중치 8*2 +bias 2, 총 18

은닉: 16 출력: (4, 2) 파라미터: 82
[레이어1] 가중치 2*16 + bias 16, 총 48
[레이어2] 가중치 16*2 + bias 2, 총 34

=> 반복되는 수치가 있으니 일반화 가능함(변수: 입력 노드, 은닉 노드, 출력 노드)
"""