# from fastapi import APIRouter, Request

# from app import logger
# from app.services.analyzer import analyze

# app = APIRouter()


# @app.post("/kg")
# async def analyze_python(github_url: str, request: Request,
#     branch: str = 'main'):
#   logger.info(f"/api/v1/persist/kg?github_url={github_url}&branch={branch}")
#   return analyze(github_url, branch, request)

from flask import Blueprint, request, jsonify
from app import logger
from app.services.analyzer import analyze

analyzer_blueprint = Blueprint('analyzer', __name__)

@analyzer_blueprint.route('/kg', methods=['POST'])
def analyze_python():
    github_url = request.args.get('github_url')
    branch = request.args.get('branch', 'main')
    
    logger.info(f"/api/v1/kg?github_url={github_url}&branch={branch}")
    
    result = analyze(github_url, branch, request)
    # If result is a set, convert to list or another JSON-safe format
    safe_result = make_json_safe(result)
    return jsonify(safe_result)

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj
