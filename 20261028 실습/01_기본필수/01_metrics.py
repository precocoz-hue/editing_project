"""실습 01: 실제 실패 6건인 손계산 예제입니다. 아직 모델을 학습하지 않습니다."""
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# 행은 실제, 열은 예측입니다. 1은 좋은 결과가 아니라 관심 클래스인 '실패'입니다.
y = [0]*14 + [1]*6                              # failed = 1
pred = [0]*10 + [1]*4 + [0]*4 + [1]*2           # TN=10, FP=4, FN=4, TP=2
print("혼동행렬(행 실제 / 열 예측, 성공0·실패1)")
print(confusion_matrix(y, pred, labels=[0, 1]))
cm = confusion_matrix(y, pred, labels=[0, 1])

# TODO B01: 네 지표 함수에 실제값 y와 예측값 pred를 순서대로 넣으세요. zero_division=0은 분모가 0일 때 0으로 표시하는 정책입니다.
acc = accuracy_score(y, pred)
precision = precision_score(y, pred, zero_division=0)
recall = recall_score(y, pred, zero_division=0)
f1 = f1_score(y, pred, zero_division=0)

# 아래처럼 개별 계산하는 것보다 함수 사용해서 계산한 결과값이 빠르게 출력됨
# tn = cm[0,0]
# fp = cm[0,1]
# fn = cm[1,0]
# tp = cm[1,1]
# #=> tn, fp, fn, tp = cm.ravel() 로 1차원 구조로 깔끔하게 구현 가능

# acc = (tp+tn)/(tn+fp+fn+tp) # (2+10)/20
# precision = tp/(tp+fp)      # 2/(2+4)
# recall = tp/(tp+fn)         # 2/(2+4)
# f1 = 2*tp/(2*tp+fp+fn)      # 2*2/(2*2+4+4)

# TODO 끝
print(f"Accuracy={acc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
print("실제 실패 6건 중 발견: 2건")

"""
<< 결과값 >>
혼동행렬(행 실제 / 열 예측, 성공0·실패1)
[[10  4]
 [ 4  2]]
Accuracy=0.600, Precision=0.333, Recall=0.333, F1=0.333
실제 실패 6건 중 발견: 2건

"""
