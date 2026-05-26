from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import feature_cols, get_top_n_recommendations


class RecommendationRequest(BaseModel):
    user_profile: dict[str, Any]
    n: int = Field(default=5, ge=1)


app = FastAPI(title="Teemo Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend")
def recommend(payload: RecommendationRequest) -> dict[str, list[dict]]:
    missing = [col for col in feature_cols if col not in payload.user_profile]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing user_profile keys: {missing}")

    return {
        "recommendations": get_top_n_recommendations(
            user_profile=payload.user_profile,
            n=payload.n,
        )
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
