"""Mock authentication for development."""

from functools import wraps
from flask import request, jsonify


def require_firebase_auth(f):
    """Mock auth decorator for development."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # In development, allow demo tokens or bypass auth
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({"error": "No authorization header"}), 401
            
        # For demo, accept any token that starts with "Bearer"
        if auth_header.startswith('Bearer '):
            return f(*args, **kwargs)
        else:
            return jsonify({"error": "Invalid authorization format"}), 401
    
    return decorated_function
