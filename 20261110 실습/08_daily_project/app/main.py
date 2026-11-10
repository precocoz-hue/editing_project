"""파일 역할 이해·분리. 실행: python -m uvicorn main:app --reload"""
from fastapi import FastAPI, HTTPException, Response
from storage import LOCK, read_data, write_data

from pydantic import BaseModel, Field

# 입력 모델 규칙 변경 (KnowledgeWrite -> RecommendWrite)
class RecommendWrite(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=2000)
    category: str = Field(min_length=1, max_length=30)

# 출력 모델 규칙 변경 (Knowledge -> Recommend)
class Recommend(RecommendWrite):
    id: int

app = FastAPI(title="신촌 맛집 정리 API", description="마음에 들었던 신촌 맛집을 관리하는 저장소")

@app.get("/health")
def health():
    return {"status": "ok"}

# API 경로 변경 (/api/knowledge -> /api/recommend)
@app.get("/api/recommend")
def list_recommends():
    with LOCK:
        data = read_data()
    return data["entries"]

# 매개변수 명칭 변경 (knowledge_id -> recommend_id)
@app.get("/api/recommend/{recommend_id}")
def get_recommend(recommend_id: int):
    with LOCK:
        data = read_data()
    for entry in data["entries"]:
        same_id = entry["id"] == recommend_id
        if same_id:
            return entry
    raise HTTPException(404, "Recommend not found")

@app.post("/api/recommend", status_code=201, response_model=Recommend)
def create_recommend(payload: RecommendWrite):
    with LOCK:
        data = read_data()
        entry = payload.model_dump()
        entry["id"] = data["next_id"]
        data["entries"].append(entry)
        data["next_id"] += 1
        write_data(data)
    return entry

@app.put("/api/recommend/{recommend_id}", response_model=Recommend)
def update_recommend(recommend_id: int, payload: RecommendWrite):
    with LOCK:
        data = read_data()
        for i, entry in enumerate(data["entries"]):
            same_id = entry["id"] == recommend_id
            if same_id:
                updated = payload.model_dump()
                updated["id"] = recommend_id
                data["entries"][i] = updated
                write_data(data)
                return updated
    raise HTTPException(404, "Recommend not found")

@app.delete("/api/recommend/{recommend_id}", status_code=204)
def delete_recommend(recommend_id: int):
    with LOCK:
        data = read_data()
        for i, entry in enumerate(data["entries"]):
            if entry["id"] == recommend_id:
                data["entries"].pop(i)
                write_data(data)
                return Response(status_code=204)
    raise HTTPException(404, "Recommend not found")
