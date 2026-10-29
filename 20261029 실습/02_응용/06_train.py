"""실습 06: 80 epoch 학습하고 10 epoch마다 같은 시점의 지표를 기록합니다."""

from pathlib import Path
import csv
import json
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
#--------------------------------------------------------------------
# 내용: train/valid 데이터 분리 + mini batch 학습 루프가 적용된 딥러닝 코드
# 목표: 주어진 2차원 좌표(x1, x2) 데이터를 보고, 0번 그룹인지 1번 그룹인지
#      가장 정확하게 맞히는 최적의 모델(가중치)과 그 학습을 기록하는 것
#--------------------------------------------------------------------
base = Path(__file__).resolve().parent
out = base / "outputs"
out.mkdir(exist_ok=True)
with (base / "data" / "points.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]
X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])
y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)
# test 40건은 모델에 전달하지 않습니다. 순서를 바꾸거나 다시 분할하지 않습니다.
torch.manual_seed(42)

# 연산 속도 극대화 by using GPU memory
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#--------------------------------------------------------------------
# 데이터 로더, mini batch 학습
#--------------------------------------------------------------------
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(43), num_workers=0)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
Xv, yv = X_valid.to(device), y_valid.to(device)
Xt, yt = X_train.to(device), y_train.to(device)
history = []

#--------------------------------------------------------------------
# 
for epoch in range(81):
    if epoch > 0:
        #--------------아무것도 배우지 않은 초기 상태의 오차와 정확도 기록
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            # 현재 batch의 loss로 역전파하고 optimizer를 한 번 갱신하세요.
            # zero_grad와 순전파·손실 계산은 위에 제공되어 있습니다.
            # >>> A06
            # TODO A06: 위 요구사항을 읽고 이 블록을 완성하세요.
            loss.backward()
            optimizer.step()
            # <<< A06
    # epoch=0은 학습 전입니다. 로그는 갱신이 끝난 동일 모델로 측정합니다.

    if epoch % 10 == 0:
        #--------------10회마다 검증시 model.eval()과 withtorch.no_grad() 선언
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(Xt), yt).item()
            logits = model(Xv)
            valid_loss = loss_fn(logits, yv).item()
            accuracy = (logits.argmax(dim=1) == yv).float().mean().item()
        history.append([epoch, train_loss, valid_loss, accuracy])
        print(epoch, round(train_loss, 6), round(valid_loss, 6), round(accuracy, 4))

#--------------------------------------------------------------------
# 결과물 도출
# 1. training_history.csv: 에포크 횟수 증가에 따른 모델의 성능 변화를 기록한 것. 과적합 발생 시점 찾기 위한 추적 리스트.
# 2. 최적화된 모델: train 학습 데이터 훈련 후 valid 데이터까지 높은 accuracy로 맞히는 인공신경망
#--------------------------------------------------------------------
with (out / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss_eval", "valid_loss", "valid_accuracy"])
    writer.writerows(history)

"""
<< 결과값 >>
0 0.565832 0.510637 0.7
10 0.331469 0.281476 0.975
20 0.228078 0.182579 0.975
30 0.176405 0.134341 1.0
40 0.145632 0.105641 1.0
50 0.12504 0.086818 1.0
60 0.110228 0.074174 1.0
70 0.098804 0.064484 1.0
80 0.089844 0.057217 1.0
---------------------------------------------------------------
# 해석
값 순서: epoch횟수  train_loss_eval    valid_loss  valid_accuracy
- train_loss_eval: 점점 우하향하며 줄어듦
- valid_loss: 줄어들다가 다시 커짐 
- valid_accuracy: 정체되는 순간 O

# 결론
몇 번째 epoch에서 과적합이 오니 학습시켜야 하는 횟수를 정할 수 있는 모델.
모델이 단순해서 과적합이 일어나지 않고 학습이 완벽하게 된 상태.

# 다른 관점
1. 에포크 횟수를 늘림
2. 모델 크기를 키움 ex.(nn.Linear(2,512),nn.Linear(512,512),nn.Linear(512.2))
"""
