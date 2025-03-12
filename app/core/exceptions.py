from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging

# Define the custom exception
class ApiException(Exception):
    def __init__(self, message: str, status_code: int = 500, detail: str = None):
        self.message = message
        self.detail = detail
        self.status_code = status_code

# Create the FastAPI application
app = FastAPI()

# Global exception handler for ApiException
@app.exception_handler(ApiException)
async def api_exception_handler(request: Request, exc: ApiException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message" : exc.message,
            "detail": exc.detail
        },
    )