import logging
import os
from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS


def create_app() -> Flask:
    """Application factory for the VocalScan backend."""
    # Load environment from a local .env if present
    load_dotenv()

    app = Flask(__name__)

    # Logging configuration
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # CORS configuration
    allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
    CORS(app, resources={r"/*": {"origins": allowed_origins}})

    @app.get("/health")
    def health() -> tuple[dict, int]:
        return jsonify({"status": "ok"}), 200

    # Register blueprints
    from .infer import infer_bp  # imported here to avoid circular imports

    app.register_blueprint(infer_bp)

    logger.info("Configured CORS for origins: %s", allowed_origins)

    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    application = create_app()
    application.run(host="0.0.0.0", port=port)


