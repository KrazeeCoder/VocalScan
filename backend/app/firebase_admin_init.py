import logging
import os
from pathlib import Path
from typing import Optional

import firebase_admin
from firebase_admin import credentials


def _detect_service_account_path() -> Optional[str]:
    """Attempt to locate a service account JSON in known locations.

    Order of precedence:
    1) GOOGLE_APPLICATION_CREDENTIALS env var
    2) FIREBASE_SERVICE_ACCOUNT_FILE env var
    3) A well-known file checked into the repo root (vocalscan-firebase-adminsdk-*.json)
    """
    env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("FIREBASE_SERVICE_ACCOUNT_FILE")
    if env_path and Path(env_path).exists():
        return env_path

    # Repo root relative to this file: backend/app/ -> go two levels up
    repo_root = Path(__file__).resolve().parents[2]

    # Prefer the explicit file name present in the repo
    candidate = repo_root / "vocalscan-firebase-adminsdk-fbsvc-2e2615b477.json"
    if candidate.exists():
        return str(candidate)

    # Fallback: first service account json found in repo root
    for json_path in repo_root.glob("*.json"):
        if "firebase" in json_path.name and json_path.is_file():
            return str(json_path)

    return None


def ensure_firebase_initialized(logger: Optional[logging.Logger] = None) -> None:
    """Initialize Firebase Admin SDK once per process."""
    if firebase_admin._apps:  # type: ignore[attr-defined]
        return

    service_account_path = _detect_service_account_path()
    if service_account_path and Path(service_account_path).exists():
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
        if logger:
            logger.info("Initialized Firebase Admin with service account at %s", service_account_path)
    else:
        # Try application default credentials (ADC)
        try:
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred)
            if logger:
                logger.info("Initialized Firebase Admin with Application Default Credentials")
        except Exception as exc:  # pragma: no cover - log-only path
            if logger:
                logger.error("Failed to initialize Firebase Admin: %s", exc)
            raise


