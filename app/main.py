# # from contextlib import asynccontextmanager

# # from fastapi import FastAPI
# # from fastapi.middleware.cors import CORSMiddleware

# # from app import settings
# # from app.core.exceptions import ApiException, api_exception_handler
# # from app.core.telemetry import setup_telemetry
# # from app.routers import analyzer_router


# # @asynccontextmanager
# # async def lifespan(context):
# #     # Setup
# #     setup_telemetry()
# #     yield
# #     # Cleanup
# #     pass

# # def create_app() -> FastAPI:

# #     app = FastAPI(
# #         title="i2-bridge",
# #         version=settings.VERSION,
# #         description="Integration bridge service",
# #         docs_url=f"{settings.API_V1_STR}/docs",
# #         openapi_url=f"{settings.API_V1_STR}/openapi.json",
# #         lifespan=lifespan,
# #         root_path=settings.API_V1_STR
# #     )

# #     # CORS
# #     app.add_middleware(
# #         CORSMiddleware,
# #         allow_origins=["*"],
# #         allow_credentials=True,
# #         allow_methods=["*"],
# #         allow_headers=["*"],
# #     )

# #     app.include_router(
# #         analyzer_router.app,
# #         tags=["kg"]
# #     )

# #     # Register global exception handlers
# #     app.add_exception_handler(
# #         ApiException,
# #         api_exception_handler)

# #     return app

# # app = create_app() 

# from flask import Flask, request, jsonify
# # from flask_cors import CORS

# from app import settings
# from app.core.exceptions import ApiException
# from app.core.telemetry import setup_telemetry
# from app.routers.analyzer_router import analyzer_blueprint


# def create_app():
#     app = Flask(__name__)
    
#     # Configuration
#     app.config['VERSION'] = settings.VERSION
#     app.config['API_PREFIX'] = settings.API_V1_STR

#     # Setup telemetry
#     setup_telemetry()

#     # Enable CORS
#     # CORS(app, supports_credentials=True)

#     # Register blueprints
#     app.register_blueprint(analyzer_blueprint, url_prefix=f"{settings.API_V1_STR}")

#     # Register error handler
#     @app.errorhandler(ApiException)
#     def handle_api_exception(error):
#         response = jsonify(error)
#         response.status_code = error.status_code
#         return response

#     return app


# # existing code ...

# app = create_app()

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8002, debug=True)
