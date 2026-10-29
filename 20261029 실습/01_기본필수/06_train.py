"""실습 06: 작은 정상 자료로 학습 순서만 익힙니다. 여기는 train 오차 확인용입니다."""
import torch
from torch import nn

torch.manual_seed(42)
X = torch.tensor([[-2., -1.], 
                  [-1., -1.], 
                  [-1., 0.], 
                  [0., -1.],
                  [1., 0.], 
                  [0., 1.], 
                  [1., 1.], 
                  [2., 1.]])
y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)    #형변환
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
print("학습 전 loss:", round(loss_fn(model(X), y).item(), 6))

model.train()       #-----------------------모델 학습을 위해 셋팅한 것
for epoch in range(200):
    # 1. zero_grad / 기울기 초기화
    optimizer.zero_grad()           #-------이전 gradient 리셋

    # 2. 순전파 및 손실 계산
    loss = loss_fn(model(X), y)     #-------모델 점수 매기기 + 틀린 오차 확인
                                    #-------현재 가진 가중치로 문제 풀고 logits 출력. 이를 실제 정답 y와 비교하여 loss 계산.
    
    # TODO B06:
    # 현재 loss의 gradient를 계산하세요.
    # 파라미터 값은 아직 바뀌지 않으며 .grad에 기울기가 들어갑니다.
    # 위 요구사항을 읽고 이 블록을 완성하세요.

    # 3. 역전파
    loss.backward()             #-----------틀린 원인 분석. pytorch가 역전파(backpropagation) 수학적 연산 수행.
                                #-----------역전파: 거꾸로 계산하며 구체적인 오차 지분율(gradient, 기울기) 계산.
                                #-----------### 아직 parameter 가중치가 바뀐 것은 아님 ###
    
    # TODO B07: 
    # 현재 gradient를 사용해 파라미터를 한 번 갱신하세요.
    # optimizer의 step 메서드를 호출합니다.
    # 위 요구사항을 읽고 이 블록을 완성하세요.

    # 4. parameter 갱신
    optimizer.step()            #-----------분석 내용을 바탕으로 parameter(가중치) 수정

# 5. 모델 평가모드 전환
model.eval()
with torch.no_grad():
    logits = model(X)
    print("학습 후 loss:", round(loss_fn(logits, y).item(), 6))
    print("train 정확도:", round((logits.argmax(dim=1) == y).float().mean().item(), 4))
print("주의: 위 정확도는 학습한 8건의 결과이며 새로운 자료의 성능이 아닙니다.")
"""
<< 결과값 >> - 80번 시도
학습 전 loss: 0.473224
학습 후 loss: 0.106799
train 정확도: 1.0
주의: 위 정확도는 학습한 8건의 결과이며 새로운 자료의 성능이 아닙니다.
-------------------------------------------------------------
<< 결과값 >> - 120번 시도
학습 전 loss: 0.473224
학습 후 loss: 0.070427
train 정확도: 1.0
주의: 위 정확도는 학습한 8건의 결과이며 새로운 자료의 성능이 아닙니다.

=> 과적합 확인을 하려면 학습에 사용되지 않은 새로운 valid set을 넣어야함.

"""