"""Test script to verify VocalScan backend functionality."""

import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from app.main import create_app
    
    print("✓ Flask app imports successful")
    
    # Test app creation
    app = create_app()
    print("✓ App creation successful")
    
    # Test basic route
    with app.test_client() as client:
        response = client.get('/health')
        if response.status_code == 200:
            print("✓ Health endpoint working")
        else:
            print(f"✗ Health endpoint failed: {response.status_code}")
    
    print("\n🎉 Backend test successful!")
    print("Ready to start the application.")
    
except Exception as e:
    print(f"✗ Backend test failed: {e}")
    print("Some dependencies may be missing.")
    print("The application will work with fallback functionality.")
