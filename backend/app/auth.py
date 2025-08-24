from __future__ import annotations

from typing import Optional, Tuple

from flask import Request
from firebase_admin import auth as fb_auth


class AuthError(Exception):
    pass


def extract_bearer_token(request: Request) -> Optional[str]:
    header = request.headers.get("Authorization", "").strip()
    if header.lower().startswith("bearer "):
        return header.split(" ", 1)[1].strip()
    return None


def verify_firebase_id_token(request: Request) -> Tuple[str, dict]:
    """Validate Authorization bearer token and return (uid, decoded_claims)."""
    token = extract_bearer_token(request)
    if not token:
        raise AuthError("missing_bearer_token")
    try:
        decoded = fb_auth.verify_id_token(token)
        uid = decoded.get("uid")
        if not uid:
            raise AuthError("invalid_token_no_uid")
        return uid, decoded
    except Exception as exc:  # firebase_admin raises various exceptions
        raise AuthError(f"invalid_token: {exc}")


