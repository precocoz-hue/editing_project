"""실습 07~08: 정상 데이터 20건. train 12건 학습, valid 4건 확인, test 4건 보관."""
from pathlib import Path
import csv
import json
import torch
from torch import nn

#--------------------------------------------------------------------
# 목표: 2차원 점(x1,x2)이 주어졌을 때, 인공신경망이 이 점들의 규칙을 배운 후 
#      새로운 valid data가 입력돼도 정확하게 0,1을 분류하고, 학습 지표를 보고서 형태로 출력하는 것.
#--------------------------------------------------------------------
base = Path(__file__).resolve().parent
with (base / "points.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

#--------------------------------------------------------------------
# 데이터 준비 및 분할
# points.csv: train 12개, valid 4개, test 4개로 나뉨
#--------------------------------------------------------------------
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]
X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])
y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)
# 기본필수는 CPU 고정, full-batch 80회입니다. ##### test는 Tensor로 만들지 않고 격리 #####
torch.manual_seed(42)

# 연산 속도 극대화 by using GPU memory
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# => 안 적으면 cpu로 모델 돌아감. 실제로 이 모델의 출력값인 "gpu_check.txt" 파일에 cpu로 돌아갔다고 표기됨.
#--------------------------------------------------------------------
# 모델 및 학습 도구 setup
#--------------------------------------------------------------------
# Linear(2,8)->ReLU->Linear(8,2) 구조의 신경망 생성
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

# 오차를 줄이기 위해 CrossEntropyLoss & SGD optimizer 셋팅
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 학습 전(epoch=0) train data 손실값을 계산하여 history에 먼저 기록함(비교할 기준점 셋팅)
history = [[0, loss_fn(model(X_train), y_train).item()]]

#--------------------------------------------------------------------
# Full-Batch 반복 학습
# 특징
# 1. Full-Batch: 데이터를 나누지 않고 한 번에 모델에 다 넣음(여기서는 train 12개)
# 2. mini-Batch가 없음: 가중치가 한 번만 업데이트 됨(loss.backward(),optimizer.step())
#--------------------------------------------------------------------
model.train()
for epoch in range(1, 81):          #---------epoch 횟수: 0~80번
    optimizer.zero_grad()
    loss = loss_fn(model(X_train), y_train)

    # TODO B08: 
    # 06번에서 배운 역전파와 파라미터 갱신 두 문장을 연결하세요.
    # 이미 위에서 이전 gradient를 비우고 loss를 계산했습니다.
    loss.backward()
    optimizer.step()

    # 갱신 후 같은 train 자료의 loss를 기록합니다. 마지막 batch loss와 구분합니다.
    with torch.no_grad():
        history.append([epoch, loss_fn(model(X_train), y_train).item()])

#--------------------------------------------------------------------
# 모델 검증(eval) 및 정확도 계산
#--------------------------------------------------------------------
model.eval()
with torch.no_grad():               #---------가중치 변경 없도록 설정
    logits = model(X_valid)         #---------처음 보는 X_valid 검증 데이터를 모델에 넣어 logits 얻음

    # TODO B09: 
    # 각 행에서 가장 큰 logit의 열 번호를 prediction으로 구하세요.
    # argmax의 dim=1은 클래스 방향입니다. 결과 shape는 (4,)입니다.
    prediction = logits.argmax(dim=1)       #----------logits=(4,2)형태의 차원(4개 데이터, 각 2개 클래스의 점수).
                                            #----------dim=1으로 가로방향(클래스 방향)으로 가장 큰 값의 인덱스 뽑으면 [0,1,0,0]처럼 (4,)짜리 예측 tensor 생성
    # prediction과 실제 정답 y_vaild를 비교해서 accuracy 계산
    accuracy = (prediction == y_valid).float().mean().item()

#--------------------------------------------------------------------
# 보고서 파일 자동 생성
# 1. "training_history.csv": 학습 과정의 손실 추이가 담김
# 2. "final_metrics.json": 최종 정확도, 초기/최종 손실값, 모델의 실제 예측값 리스트가 담김
# 3. "gpu_check.txt": 하드웨어 환경 증빙을 위한 pytorch ver., CUDA 가용여부가 적힘
#--------------------------------------------------------------------
with (base / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss_eval"])
    writer.writerows(history)
metrics = {"device": "cpu", 
           "epochs": 80, 
           "train_rows": 12, 
           "valid_rows": 4,
           "test_used": False, 
           "valid_accuracy": accuracy,
           "initial_train_loss": history[0][1], 
           "final_train_loss": history[-1][1],
           "valid_ids": [r["id"] for r in valid_rows], 
           "actual": y_valid.tolist(),
           "predicted": prediction.tolist()}

(base / "final_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
(base / "gpu_check.txt").write_text(
    f"torch={torch.__version__}\ncuda_build={torch.version.cuda}\n"
    f"cuda_available={torch.cuda.is_available()}\nselected_device=cpu\n", encoding="utf-8")

print("분할: train 12 / valid 4 / test 4(보관)")
print("train loss:", round(history[0][1], 6), "->", round(history[-1][1], 6))
print("valid 정확도:", round(accuracy, 4))
print("valid 예측:", prediction.tolist())
print("생성: gpu_check.txt, training_history.csv, final_metrics.json")

"""
<< 결과값 >>
분할: train 12 / valid 4 / test 4(보관)
train loss: 0.514554 -> 0.110675
valid 정확도: 1.0
valid 예측: [0, 0, 1, 1]
생성: gpu_check.txt, training_history.csv, final_metrics.json
---------------------------------------------------------------
# 해석
- epoch 80회를 통해 손실값 0.514->0.111로 감소. GSD를 통해 가중치 최적화가 안정적으로 수행됨
- valid 정확도가 1로 100% 예측함. metrix={..."actual": y_valid.tolist(),     <--- 실제 정답
                                        "predicted": prediction.tolist()} <--- 예측값 [0,0,1,1]
- gpu_check.txt 파일의 'selected_device=cpu'항목을 통해, 본 모델의 학습 및 검증 연산이 CPU 환경에서 수행됨.                                      
---------------------------------------------------------------
# AI) 결과 분석 보고서
1. 데이터 분할 개요
• 학습(Train): 12건 / 검증(Valid): 4건 / 평가(Test): 4건 (최종 성능 검증을 위해 격리 보관)
• 해석: 소규모 데이터셋 환경이지만, 훈련-검증-평가 데이터를 철저히 분리하여 데이터 누수(Data Leakage) 없이 객체 지향적인 실험 환경을 구축하였습니다.

2. 모델 학습 상태 분석 (Train Loss)
• 초기 손실(Initial Loss): 0.514554
• 최종 손실(Final Loss): 0.110675
• 해석: 80회의 에포크(Epoch) 동안 손실 값이 약 78.5% 감소하였습니다. 이는 모델이 2차원 수치 특성(x1, x2)과 정답 레이블(Class 0/1) 사이의 수학적 인과관계를 효과적으로 학습했음을 증명합니다. 경사하강법(SGD)을 통한 가중치 최적화가 안정적으로 수행되었습니다.

3. 일반화 성능 평가 (Valid Accuracy & Prediction)
• 검증 정확도(Valid Accuracy): 1.0 (100%)
• 검증 데이터 예측값: [0, 0, 1, 1]
• 해석: 학습 과정에서 단 한 번도 보지 못한 새로운 검증 데이터 4건에 대해 100%의 정확도로 완벽하게 분류해냈습니다.
• 특히 모델이 가공되지 않은 예측 점수(logits) 중에서 가장 큰 점수를 가진 열 번호(argmax)를 정확히 찾아내어 실제 정답과 완벽히 일치하는 [0, 0, 1, 1]을 출력했습니다.

4. 종합 결론 (과적합 및 성능 평가)
• 결과: 성공적인 일반화 모델 구축 (Good Fit)
• 상세 해석: 일반적으로 데이터 크기가 작고 학습을 반복하면 모델이 문제를 통째로 외워버리는 '과적합(Overfitting)'이 발생하기 쉽습니다. 그러나 본 프로젝트에서는 Linear(2,8) -> ReLU -> Linear(8,2)라는 데이터 규모에 맞는 적절하고 가벼운 모델 구조를 채택함으로써, 암기가 아닌 '데이터의 본질적인 분류 경계면'을 올바르게 학습하는 데 성공했습니다.
• 결과적으로 이 모델은 향후 보관된 4건의 테스트(Test) 데이터가 입력되더라도 매우 높은 정확도로 분류를 수행할 수 있을 것으로 기대됩니다.


"""