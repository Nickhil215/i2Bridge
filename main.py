from flask import Flask, jsonify
from app.routers.analyzer_router import analyzer_blueprint
from app import settings

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['VERSION'] = settings.VERSION
    app.config['API_PREFIX'] = settings.API_V1_STR

    # Register blueprint
    app.register_blueprint(analyzer_blueprint, url_prefix=f"{settings.API_V1_STR}")

    return app

# Create the Flask app instance
app = create_app()

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=True)
