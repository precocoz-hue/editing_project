from pathlib import Path
import pandas as pd
import torch
from torch import nn

# 내용: 서로 다른 두 가지 학습률(0.001, 0.01) 조건에서 인공신경망 모델을 각각 20 epoch 동안 학습시키고 
#      성능 추이를 비교 분석함
# 요약: hyperparameter tuning code
#      [1] lr=0.001 (baseline) 조건으로 모델을 생성하여 20 epoch 동안 순전파, 역전파, 오차 기록 후 결과를 CSV 및 그래프 파일로 저장함
#      [2] 난수 시드를 재고정하여 완전히 초기화된 상태에서 lr=0.01 (candidate) 조건으로 20 epoch 동안 동일한 과정을 반복함
#      [3] 두 결과물의 검증 손실(valid_loss) 추이를 대조하여 이 데이터셋에 더 빠르고 효과적으로 수렴하는 최적의 학습률 파라미터를 선정함
# -------------------------------------------------------------------------------
torch.set_num_threads(1)
# CPU 연산 스레드 수를 1개로 제한함
# 멀티스레딩 오버헤드를 방지하고 시스템 사양에 관계없이 동일한 연산 속도와 재현 가능한 학습 결과를 보장하기 위함
# -------------------------------------------------------------------------------
base = Path(__file__).resolve().parent
# 현재 실행 중인 파이썬 스크립트 파일(__file__)의 절대 경로를 구한 뒤 부모 폴더 경로를 기준 경로(base)로 설정함
# -------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt
# 백그라운드 환경에서 GUI 창 팝업 없이 그래프를 PNG 이미지 파일로 직접 저장하기 위해 matplotlib의 렌더링 엔진을 Agg로 선언함
# -------------------------------------------------------------------------------
df = pd.read_csv(base / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
# base 경로에서 'points.csv' 파일을 읽어와 Pandas 데이터프레임으로 변환함
# 'split' 열의 텍스트가 'train'인 행들은 학습셋, 'valid'인 행들은 검증셋으로 필터링하여 분리함
# -------------------------------------------------------------------------------
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long)
# 분리된 데이터프레임의 값(.values)을 pytorch 연산용 텐서(tensor) 배열로 생성함
# 입력 특성(x1, x2)은 모델 연산이 가능하도록 실수형(float32)을 지정하고, 정답 레이블(label)은 다중 분류용 인덱스 정수형(long) 타입을 부여함
# -------------------------------------------------------------------------------
loss_fn = nn.CrossEntropyLoss()
# 모델의 클래스별 예측 확률 분포와 실제 정답 원-핫 레이블 간의 괴리율을 측정하는 교차 엔트로피 손실 함수 객체를 생성함
# -------------------------------------------------------------------------------
summary = []
# 각각의 학습률 실험 조건에서 도출된 최적의 스펙 정보들을 취합하여 통합 보관하기 위한 빈 리스트를 생성함
# -------------------------------------------------------------------------------
for name, lr in [("baseline", 0.001), ("candidate", 0.01)]:
    # hyper parameter 튜닝을 위해 실험 조건 명칭(name)과 매핑된 학습률 값(lr)을 튜플 형태로 공급받아 순회하는 메인 루프 구역

    torch.manual_seed(42)
    # 다른 학습률 조건이더라도 초기 가중치의 상태를 완전히 똑같이 맞추기 위해 pytorch 글로벌 랜덤 시드를 42로 고정

    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    # nn.Sequential: 입력된 레이어들을 순차적으로 통과시키는 컨테이너 구조 신경망을 정의함
    # nn.Linear(2, 8): 2개의 입력 변수(x1, x2)를 받아 은닉층 노드 8개로 사상(Mapping)하는 선형 결합 레이어
    # nn.ReLU(): 음수 출력을 0으로 차단하여 모델에 비선형 표현력을 부여하는 활성화 함수 레이어
    # nn.Linear(8, 2): 은닉 노드 8개의 신호를 취합하여 최종 분류 클래스 개수인 2개의 원본 점수(Logits)로 출력하는 레이어
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    # model.parameters(): 옵티마이저가 역전파 기울기를 보고 갱신해야 할 대상인 모델의 가중치와 편향 파라미터들을 통째로 전달함
    # lr=lr: 루프에서 지정된 보폭 계수(0.001 또는 0.01)를 대입하여 가중치 이동 거리를 결정함
    # weight_decay=0.0: 과적합 방지용 가중치 감쇠(벌금 규제) 강도를 0으로 두어 순수한 오차 손실만 단속하도록 세팅함

    history = []
    # 현재 조건의 루프 안에서 매 epoch마다 연산되는 세부 학습 통계량을 순차적으로 누적 보관할 빈 리스트를 생성함

    for epoch in range(1, 21):
        # 1 epoch부터 시작하여 설정된 제안 기준인 총 20 epoch 동안 모델 학습 및 평가를 주도하는 서브 루프 구역

        # ------------------------------ 학습 ------------------------------
        model.train()
        # 학습 가동 상태로 모델의 내부 모드를 설정함 (DropOut이나 BatchNorm 등의 학습용 로직을 켜는 약속된 필수 선언문임)

        optimizer.zero_grad()
        # 가중치 갱신 연산 전에 직전 epoch 학습으로 인해 모델 파라미터 각 노드에 누적되어 남아있던 미분 기울기 메모리를 영(0)으로 청소함

        loss = loss_fn(model(X_train), y_train)
        # model(X_train): 전체 학습 입력을 모델에 주입하여 예측 원본 점수인 로짓(Logits) 배열을 뽑아내는 순전파를 수행함
        # loss_fn(..., y_train): 도출된 로짓 점수와 정답 레이블을 교차 엔트로피 수식에 대입하여 현재 모델의 총 평균 오차값을 도출함

        # >>> B08
        # TODO B08: 요구사항
        loss.backward()
        # 오차 역전파 실행: 산출된 오차(loss)를 기점으로 출력층에서 입력층 방향으로 합성함수 미분(연쇄법칙)을 수행하여 각 가중치의 미분 기울기를 구함

        optimizer.step()
        # 가중치 실제 업데이트: 옵티마이저(AdamW)가 방금 구한 미분 방향과 지정된 학습률(lr) 속도를 조합하여 가중치들을 실제로 한 단계 이동시킵니다.
        # <<< B08

        # ------------------------------ 평가 ------------------------------
        model.eval()
        # 가중치 동결 및 추론 전용 상태로 모델의 내부 모드를 세팅함 (평가 시 오작동할 수 있는 학습용 레이어들의 기능을 일시 차단)

        with torch.no_grad():
            # 평가 단계에서는 역전파용 미분 그래프 생성이 불필요
            # ㄴ> 컴퓨터 미분 메모리 연산을 완전히 차단하여 속도를 가속화하는 컨텍스트 매니저를 작동시킴

            train_loss = loss_fn(model(X_train), y_train).item()
            # 갱신이 완료된 최신 가중치 모델에 다시 학습 데이터를 넣어 손실값을 구한 뒤 `.item()`으로 파이썬 순수 실수 값으로 치환하여 저장

            valid_loss = loss_fn(model(X_valid), y_valid).item()
            # 한 번도 학습에 쓰이지 않은 독립된 검증 데이터(X_valid, y_valid)를 모델에 대입하여 현재 신경망의 일반화 오차 성능을 평가함

        # >>> B09
        # TODO B09: 요구사항
        history.append([epoch, train_loss, valid_loss])
        # 현재 진행 완료된 epoch 번호(epoch), 해당 시점의 학습 데이터 손실(train_loss), 검증 데이터 손실(valid_loss)을 하나의 리스트로 묶어 이력 배열에 기록함
        # <<< B09

    # -------------------------------------------------------------------------------
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss"])
    table.to_csv(base / f"history_{name}.csv", index=False)
    # 20 epoch 루프 종료 후 축적된 기록 이력 리스트를 칼럼명이 지정된 Pandas 데이터프레임 구조로 변환하고 기준 경로에 CSV 파일로 저장함

    best = table.loc[table["valid_loss"].idxmin()]
    # 전체 20개 epoch 중에서 검증 오차(valid_loss) 항목이 가장 최소값(최적 성능)을 기록한 지점의 행 인덱스를 찾아 해당 데이터를 통째로 반환

    summary.append([name,                   # 현재 실험 명칭
                    lr,                     # 적용 학습률
                    int(best["epoch"]),     # 최적 성능을 낸 epoch 번호
                    best["valid_loss"]])    # 최저 검증 오차 수치
    # 종합 비교용 결과 리스트에 위 항목들을 리스트 형태로 추가

    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval mode)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base / f"07_curves_{name}.png", dpi=150)
    plt.close()
    # epoch 진행 흐름에 따른 train 및 valid 손실 곡선을 차트로 그리고, 가장 이상적인 오차를 나타낸 지점에 수직 점선을 작도하여 PNG 그림 파일로 영구 저장
# -------------------------------------------------------------------------------
result = pd.DataFrame(summary, columns=["setting", "lr", "best_epoch", "best_valid_loss"])
result.to_csv(base / "experiment_table.csv", index=False)
# 취합된 실험 요약 2차원 리스트를 통합 데이터프레임 구조로 변경하고 전체 모델 간 결과 비교용 통합 CSV 파일로 내보냄

print(result.round(6).to_string(index=False))
print("test 미사용 / best epoch는 관찰한 최소 지점, 기본필수에서는 모델 저장·복원하지 않음")
# 최종 대조 지표 테이블 정보를 소수점 6자리까지 정밀 반올림하여 콘솔창에 문자열 형태로 깔끔하게 출력하고 마침 고지 안내문을 표기함

"""
<< 결과값 >>
  setting    lr  best_epoch  best_valid_loss
 baseline 0.001          20         0.480205
candidate 0.010          20         0.307332
test 미사용 / best epoch는 관찰한 최소 지점, 기본필수에서는 모델 저장·복원하지 않음

-------------------------------------------------------------------------------
# 해석
1. best_valid_loss 비교
- baseline      0.480205
- candidate     0.307332
=> Loss는 0에 가까울수록 정답을 더 잘 맞힌다는 의미이므로, lr(학습률)을 0.01로 설정한 candidate 모델이 baseline 보다 성능이 우수함

2. best_epoch 비교
- 둘다 20으로 동일. 둘다 검증 손실이 우하향하며 안정화 되었기 떄문
- epoch 제한 수를 더 늘려서 돌려보고 검증하는 것이 좋음

# 결론
1. lr(학습률)을 0.001 보다는 0.01로 설정하는 것이 20번의 한정된 epoch 내에서 빠르게 정답에 도달할 수 있음

"""