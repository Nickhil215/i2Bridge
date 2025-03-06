from fastapi import APIRouter, Request
from app.services import logger
from app.services.analyzer import analyze

app = APIRouter()

@app.get("/kg")
def analyze_python(github_url: str, request: Request):
  logger.info(f"/api/v1/persist/kg?github_url={github_url}")
  analyze(github_url, request)
  return {"msg" : "success"}