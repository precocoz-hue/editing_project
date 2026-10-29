"""실습 05: 정답 인덱스와 logits로 손실을 계산합니다."""
import torch
from torch import nn

logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]]) #--------모델이 예측한 점수 Logits. 행이 2개 -> 배치크기 2
target = torch.tensor([0, 1], dtype=torch.long) #--------실제 정답 Target.
loss_fn = nn.CrossEntropyLoss()                 #--------정답을 torch.long(정수형 데이터)타입으로 넣기
                                                #--------내부적으로 [1]LogSoftmax [2]NLLLoss 를 한 번에 계산하도록 설계됨
# loss_fn에 logits와 target을 순서대로 넣어 loss를 계산하세요.
# logits에 softmax를 먼저 적용하지 마세요.
# >>> B05
# TODO B05: 위 요구사항을 읽고 이 블록을 완성하세요.
loss = torch.tensor(loss_fn(logits,target))     #--------배치 크기가 2->각 손실값의 평균값 = loss
    #= loss_fn(logits,target)으로 작성해도 무방함
# <<< B05
print("logits:", tuple(logits.shape))
print("target:", tuple(target.shape), target.dtype)
print("loss:", round(loss.item(), 6))
"""
<< 결과값 >>
logits: (2, 2)              <--- 2개 데이터, 각 2개 클래스 점수
target: (2,) torch.int64    <--- 정답 레이블 2개
loss: 0.126928              <--- 소수점 6자리까지 반올림한 손실값

# 계산 방법
1. Softmax 변환 (내부 동작)
- [2.0,0.0], [0.0,2.0]에 소프트맥스 적용시 확률은 [0.88,0.12],[0.12,0.88].
2. 정답 확률 추출 및 로그 계산
첫번째 데이터의 정답(0)인 확률: log(0.88)=0.126928
두번째 데이터의 정답(1)인 확률: log(0.88)=0.126928
3. 최종 평균
loss: 0.126928

"""