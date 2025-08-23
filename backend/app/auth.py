import json
import os
from functools import wraps
from typing import Any, Callable

from flask import jsonify, request, g
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials


def initialize_firebase_admin() -> None:
    """Initialize Firebase Admin SDK if not already initialized.

    This supports three ways to provide credentials, in priority order:
    1) GOOGLE_APPLICATION_CREDENTIALS_JSON (inline JSON string)
    2) Application Default Credentials (works with GOOGLE_APPLICATION_CREDENTIALS path)
    3) Fallback to default initialization (may work in some environments)
    """
    if firebase_admin._apps:
        return

    creds = None
    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if credentials_json:
        try:
            creds = credentials.Certificate(json.loads(credentials_json))
        except Exception:
            creds = None
    if creds is None:
        try:
            creds = credentials.ApplicationDefault()
        except Exception:
            sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if sa_path and os.path.exists(sa_path):
                try:
                    creds = credentials.Certificate(sa_path)
                except Exception:
                    creds = None
            else:
                creds = None

    if creds is not None:
        firebase_admin.initialize_app(creds)
    else:
        # Last resort: try without explicit credentials
        firebase_admin.initialize_app()


def require_firebase_auth(view_func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to require a valid Firebase ID token via Bearer auth."""

    @wraps(view_func)
    def wrapper(*args: Any, **kwargs: Any):
        initialize_firebase_admin()

        auth_header: str = request.headers.get("Authorization", "")
        prefix = "Bearer "
        if not auth_header.startswith(prefix):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        id_token = auth_header[len(prefix):].strip()
        try:
            decoded_token = firebase_auth.verify_id_token(id_token)
        except Exception:
            return jsonify({"error": "Invalid or expired token"}), 401

        g.firebase_user = decoded_token
        g.uid = decoded_token.get("uid")
        return view_func(*args, **kwargs)

    return wrapper


