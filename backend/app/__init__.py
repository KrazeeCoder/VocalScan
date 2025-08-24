import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, redirect, url_for
from flask_cors import CORS

from .firebase_admin_init import ensure_firebase_initialized


def _parse_allowed_origins(env_value: Optional[str]) -> List[str]:
    if not env_value:
        # Sensible defaults for local development
        return [
            "http://localhost:5000",
            "http://127.0.0.1:5000",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]
    return [origin.strip() for origin in env_value.split(",") if origin.strip()]


def create_app() -> Flask:
    """Application factory for the VocalScan Flask app."""
    load_dotenv()

    # Initialize Flask
    app = Flask(__name__, static_folder="../static", template_folder="../templates")

    # Logging configuration
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # CORS configuration
    allowed_origins = _parse_allowed_origins(os.getenv("ALLOWED_ORIGINS"))
    CORS(app, resources={r"/*": {"origins": allowed_origins}})

    # Firebase Admin initialization
    ensure_firebase_initialized(logger)

    # Health route
    @app.get("/health")
    def health() -> tuple[dict, int]:
        return jsonify({"status": "ok"}), 200

    # Register blueprints
    from .infer import infer_bp
    from .pages import pages_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(infer_bp)

    # Root redirects to login page (client guards will redirect post-login)
    @app.get("/")
    def index():
        return redirect(url_for("pages.login"))

    logger.info("VocalScan Flask app initialized. CORS for: %s", ", ".join(allowed_origins))
    return app


