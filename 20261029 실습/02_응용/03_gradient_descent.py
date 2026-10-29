"""실습 03: 매번 새로운 그래프를 만들고 gradient를 비웁니다."""
from pathlib import Path
import csv
import torch

# w를 목표값 4에 가깝게 10회 갱신하는 것
# no_grad 갱신과 gradient 초기화를 연결해야 됨
w = torch.tensor(0.0, requires_grad=True)
lr = 0.1
history = []
for step in range(1, 11):       #---------- 4) 10번 반복 수행
    loss = (w - 4.0) ** 2       #---------- 1) 오차(Loss) 계산. 목표값 4.0에서 얼마나 멀리 떨어져있는지 확인.
    loss.backward()             #---------- 2) 기울기 계산(역전파). Pytorch 자동 미분 기능.
    with torch.no_grad():       # 가중치 변경 시 미분 계산 멈추는 역할
        # w에서 학습률 × 현재 기울기를 빼세요.
        # 여기는 no_grad 블록 안입니다. 새 값으로 Tensor 객체를 교체하지 말고 w를 직접 갱신하세요.
    
        # >>> A03
        # TODO A03: 위 요구사항을 읽고 이 블록을 완성하세요.
        w -= lr * w.grad        #---------- 3) w 업데이트(Gradient Descent 공식)
        # w에서 학습률 * 현재 기울기를 뺀 값
        # pytorch가 제공하는 기본 최적화 도구: torch.optim.SGD
        # <<< A03

    w.grad.zero_()              #---------- 5) 기울기 리셋. pytorch는 기본적으로 새로 구한 기울기를 기존 기울기에 계속 더하는 성질 O. 
                                #----------    기존 기울기를 0으로 리셋하는 역할 필요.


    # 아래 loss는 갱신한 w에서 다시 계산한 값입니다.
    history.append([step, w.item(), (w.detach() - 4.0).square().item()])
for step, value, loss_after in history:
    print(step, round(value, 6), round(loss_after, 6))
out = Path(__file__).resolve().parent / "outputs"
out.mkdir(exist_ok=True)
with (out / "gd_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f); writer.writerow(["step", "w_after", "loss_after"]); writer.writerows(history)

"""
<< 결과값 >>
1 0.8 10.240001
2 1.44 6.5536
3 1.952 4.194304
4 2.3616 2.684354
5 2.68928 1.717987
6 2.951424 1.099511
7 3.161139 0.703687
8 3.328911 0.45036
9 3.463129 0.28823
10 3.570503 0.184467
------------------------------------------
# 해석
값 순서: step     value      loss_after
-> step 진행될수록 value 값이 4에 가까워지고 loss_after값은 0에 가까워짐(기울기가 완만해짐)
"""