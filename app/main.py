
from flask import Flask, request, jsonify
# from flask_cors import CORS

from app import settings
from app.core.exceptions import ApiException
from app.core.telemetry import setup_telemetry
from app.routers.analyzer_router import analyzer_blueprint


def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['VERSION'] = settings.VERSION
    app.config['API_PREFIX'] = settings.API_V1_STR

    # Setup telemetry
    setup_telemetry()

    # Enable CORS
    # CORS(app, supports_credentials=True)

    # Register blueprints
    app.register_blueprint(analyzer_blueprint, url_prefix=f"{settings.API_V1_STR}")

    # Register error handler
    @app.errorhandler(ApiException)
    def handle_api_exception(error):
        response = jsonify(error)
        response.status_code = error.status_code
        return response

    return app


# existing code ...

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=True)
