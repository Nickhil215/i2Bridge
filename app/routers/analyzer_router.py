from fastapi import APIRouter, Request

from app import logger
from app.services.analyzer import analyze
# import asyncio

app = APIRouter()
# lock = asyncio.Lock()

@app.post("/kg")
async def analyze_python(github_url: str, request: Request, branch: str = 'main'):
  # async with lock:
    logger.info(f"/api/v1/persist/kg?github_url={github_url}&branch={branch}")
    return analyze(github_url, branch, request)
