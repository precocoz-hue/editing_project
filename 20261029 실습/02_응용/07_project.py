"""실습 07~08: 앞 실습을 한 파일로 연결하고 환경·학습·평가 결과를 저장합니다."""

from pathlib import Path
import csv
import json
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# 1. 경로 설정 및 파일 저장 폴더(outputs) 생성
base = Path(__file__).resolve().parent
out = base / "outputs"
out.mkdir(exist_ok=True) # 폴더가 이미 있으면 그대로 두고, 없으면 새로 만듦

# 2. CSV 데이터 로드 및 분할 (Train / Valid)
with (base / "data" / "points.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f)) # CSV 데이터를 딕셔너리 형태로 읽어 리스트로 변환
train_rows = [r for r in rows if r["split"] == "train"] # 훈련 데이터 필터링
valid_rows = [r for r in rows if r["split"] == "valid"] # 검증 데이터 필터링

# 3. 데이터를 pytorch Tensor로 변환 (좌표는 실수형, 레이블은 정수형)

X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
# X_valid: points.csv에서 x1과 x2라는 두 개의 '숫자 좌표'를 꺼내서 리스트로 만든 것 (문제)
X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])
# y_valid: points.csv에서 이 점이 0번 그룹인지 1번 그룹인지 '정답 분류'를 꺼내서 만든 것 (정답)
y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)
# test 40건은 모델에 전달하지 않습니다. 순서를 바꾸거나 다시 분할하지 않습니다.

# 4. 재현성을 위한 전역 난수 시드(Seed) 설정
torch.manual_seed(42)

# 5. 연산 장치(GPU / CPU) 설정 및 데이터 로더 정의
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(43), num_workers=0)     
# manual_seed(43)은 에포크마다 train 데이터가 섞이는 무작위 순서만 독립적으로 고정

# 6. 인공신경망 모델(MLP), 손실함수(CrossEntropy), 옵티마이저(SGD) 정의 및 장치 이동
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)   # 인공신경망
loss_fn = nn.CrossEntropyLoss()                                                 # 손실함수
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)                         # SGD 옵티마이저

# 7. 전체 훈련/검증 데이터를 지정된 장치(GPU 또는 CPU)로 복사
Xt, yt = X_train.to(device), y_train.to(device)
Xv, yv = X_valid.to(device), y_valid.to(device)
history = []

# 8. 학습 루프 진행 (총 81회: Epoch 0 ~ 80)
for epoch in range(81):     #----------epoch 80번 시행
    if epoch > 0: # epoch 0은 학습 전 상태를 기록하기 위해 건너뛰고, 1부터 가중치를 학습합니다.
        model.train() # 모델을 학습 모드로 설정
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device) # 미니배치 데이터를 장치로 이동
            optimizer.zero_grad()                 # 이전 배치의 그라디언트(기울기) 초기화
            loss = loss_fn(model(xb), yb)         # 순전파(Forward) 연산 및 손실값 계산

            # TODO A07: 역전파(기울기 계산) 및 가중치 업데이트 연결
            loss.backward()  # 역전파를 수행하여 gradient 계산
            optimizer.step()  # 옵티마이저를 사용해 모델의 가중치 갱신

    # 9. 10 epoch마다 모델 성능 평가 및 기록
    if epoch % 10 == 0:     #-----------epoch 10회마다 평가
        model.eval() # 모델을 평가 모드로 설정 (드롭아웃 등이 비활성화됨)
        with torch.no_grad(): # 평가 시에는 메모리 절약을 위해 그라디언트 추적을 끕니다.
            train_loss = loss_fn(model(Xt), yt).item() # 전체 훈련 데이터 손실 계산
            logits = model(Xv)                         # 검증 데이터 순전파 수행
            valid_loss = loss_fn(logits, yv).item()    # 검증 데이터 손실 계산
            accuracy = (logits.argmax(dim=1) == yv).float().mean().item() # 검증 데이터 정확도 계산
        history.append([epoch, train_loss, valid_loss, accuracy]) # 결과를 기록 리스트에 추가
        print(epoch, round(train_loss, 6), round(valid_loss, 6), round(accuracy, 4))

# 10. 학습 기록(History)을 CSV 파일로 저장
with (out / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss_eval", "valid_loss", "valid_accuracy"]) # 헤더 작성
    writer.writerows(history) # 모든 에포크별 성능 지표 저장


# 11. 최종 모델 검증(eval) 및 최종 예측값 추출
model.eval()
with torch.no_grad():
    logits = model(Xv)

    # TODO A08: 최종 예측값 구하기
    # 각 데이터별로 확률(점수)이 가장 높은 클래스의 인덱스(0 또는 1)를 추출하고, CPU 메모리로 옮깁니다.
    predictions = logits.argmax(dim=1).cpu()

# 12. 최종 학습/환경 메타데이터 지표를 딕셔너리로 구성하여 JSON 파일로 저장
metrics = {"device": str(device), 
           "torch": str(torch.__version__), 
           "epochs": 80,
           "train_rows": len(train_rows), 
           "valid_rows": len(valid_rows), 
           "test_used": False,
           "initial_train_loss": history[0][1],  # 에포크 0의 훈련 손실
           "final_train_loss": history[-1][1],    # 최종 에포크의 훈련 손실
           "valid_loss": history[-1][2],          # 최종 에포크의 검증 손실
           "valid_accuracy": history[-1][3]}      # 최종 에포크의 검증 정확도
           
(out / "final_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

# 13. 모든 검증 데이터의 ID, 실제 정답, 모델의 예측값을 매핑하여 CSV로 저장
with (out / "validation_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f); writer.writerow(["id", "actual", "predicted"])
    # zip을 사용하여 원본 데이터, 정답 텐서, 예측 텐서를 동시에 한 행씩 순회하며 작성합니다.
    writer.writerows((r["id"], int(y), int(p)) for r, y, p in zip(valid_rows, y_valid, predictions))

# 14. 사용 중인 파이토치 및 CUDA 환경 정보를 텍스트 파일로 기록
(out / "gpu_check.txt").write_text(
    f"torch={torch.__version__}\ncuda_build={torch.version.cuda}\n"
    f"cuda_available={torch.cuda.is_available()}\nselected_device={device}\n", encoding="utf-8")

print("생성: outputs의 환경·학습·검증 결과")

# 추가 A10: valid의 실제값과 예측을 비교해 오답 목록을 만드세요.
    # 입력: valid_rows, y_valid, predictions는 동일한 40개 ID 순서입니다.
    # 반환 형식: errors=[(요청ID, 실제정답int, 예측int), ...].
    # 조건: 실제값과 예측이 다른 행만 포함하고 원래 valid 순서를 유지합니다.
    # 금지: test 자료 사용, 정답 변경, 오답이 없을 때 가상의 오답 추가.
    # 완료: 오류 행 수 = 40 - 맞힌 개수, 저장 파일은 헤더를 포함합니다.
# >>> A10
# TODO A10: 리스트 컴프리헨션을 사용해 실제 정답(y)과 예측값(p)이 일치하지 않는(오답) 행만 한 줄로 필터링하여 수집합니다.
errors = [(r["id"], int(y), int(p)) for r, y, p in zip(valid_rows, y_valid, predictions) if int(y) != int(p)]
# <<< A10

# 15. 추출된 오답 목록을 CSV 파일로 최종 저장
with (out / "validation_errors.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "actual", "predicted"]) # 헤더 작성
    writer.writerows(errors) # 오답 행들 작성
print("valid 오답 수:", len(errors))
