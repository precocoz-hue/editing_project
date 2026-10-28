"""실습 05 응용: train 수치2열만 결측 대체·표준화하여 K=3으로 묶습니다."""
from pathlib import Path
import pandas as pd
import numpy as np
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"].tolist()
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"].tolist()
train, valid = raw.loc[train_ids].copy(), raw.loc[valid_ids].copy()
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]

# NUM(Numerical features): input_chars, allowed_tools
# CAT(Categorical features): task_type, priority
NUM, CAT = FEATURES[:2], FEATURES[2:]   # python slicing: [:2] means 0,1 index, [2:] means 2,3 index
X_train, X_valid = train[FEATURES].copy(), valid[FEATURES].copy()
y_train, y_valid = train["failed"], valid["failed"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
cluster = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("kmeans", KMeans(n_clusters=3, n_init=10, random_state=42))])
# TODO A05: 수치2열만으로 cluster를 학습하세요. failed나 elapsed_ms를 넣지 않습니다.
cluster.fit(X_train[NUM])
# TODO 끝

# 목적: k-means 학습 후 각 cluster의 특징 분석하는 profile 생성 & 품질 평가
# 몇 번 군집(0,1,2)으로 데이터가 지정되었는지 정답 배열 불러오기
labels = cluster["kmeans"].labels_
# pipeline의 k-means 단계를 제외한 simpleImputer와 standardScaler 수행
z = cluster[:-1].transform(X_train[NUM])
# centers: K-means의 각 군집 중심 좌표를 inverse_transform로 원래 단위로 되돌리기
centers = cluster["scaler"].inverse_transform(cluster["kmeans"].cluster_centers_)
# profile: 복구된 중심점 데이터로 DataFrame 생성. 앞에 군집 번호 (0,1,2)추가.
profile = pd.DataFrame(centers, columns=NUM)
profile.insert(0, "cluster", range(3))
profile["n"] = np.bincount(labels, minlength=3)
#군집 통계 요약표 + 요청ID별 군집 매칭 결과 저장
profile.to_csv(OUT / "cluster_profile.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"request_id": train_ids, "cluster": labels}).to_csv(OUT / "cluster_assignments.csv", index=False, encoding="utf-8-sig")

# silhouette_score 사용 이유: 군집 개수가 3개로 고정된 상태에서 inertia(군집 내 거리 합계)만으로는 군집 품질을 평가하기 어렵기 때문.
# silouette_score: -1~1 사이의 값 출력. 1에 가까울수록 군집 품질이 좋음.
print("silhouette(표준화 공간):", round(silhouette_score(z, labels), 3))
print("군집 번호에는 실패·성공·우열의 뜻이 없습니다.")

"""
<< 결과값 >>
 cluster  input_chars  allowed_tools  n
       0     1542.250          4.125 24     <--- 0번 군집: 핵심 그룹. 글자수 제일 많고 허용도구도 많음
       1     1430.474          1.368 19     <--- 1번 군집: 글자수는 많지만 허용도구가 적음
       2      444.588          2.706 17     <--- 2번 군집: 글자수는 적지만 허용도구는 많음
silhouette(표준화 공간): 0.453               <--- 실루엣 점수: 3개 군집이 서로 겹치지 않고 타당성 있게 쪼개진 것으로 생각됨
군집 번호에는 실패·성공·우열의 뜻이 없습니다.

"""