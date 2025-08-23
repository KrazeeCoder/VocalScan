#!/usr/bin/env python3
"""Setup script for VocalScan development environment."""

import os
import sys
import subprocess
import platform

def run_command(cmd, cwd=None):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, cwd=cwd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {cmd}")
        print(f"  Error: {e.stderr}")
        return False

def setup_backend():
    """Set up the backend environment."""
    print("\n=== Setting up Backend ===")
    
    backend_dir = "backend"
    if not os.path.exists(backend_dir):
        print(f"Error: {backend_dir} directory not found")
        return False
    
    # Create virtual environment
    venv_cmd = "python -m venv venv" if platform.system() == "Windows" else "python3 -m venv venv"
    if not run_command(venv_cmd, cwd=backend_dir):
        return False
    
    # Determine activation script
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    # Install requirements
    if not run_command(f"{pip_cmd} install --upgrade pip", cwd=backend_dir):
        return False
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", cwd=backend_dir):
        return False
    
    # Train initial models
    print("Training initial ML models...")
    if not run_command(f"{python_cmd} train_models.py", cwd=backend_dir):
        print("Warning: Model training failed, will use fallback models")
    
    print("✓ Backend setup complete")
    return True

def setup_frontend():
    """Set up the frontend environment."""
    print("\n=== Setting up Frontend ===")
    
    frontend_dir = "frontend"
    if not os.path.exists(frontend_dir):
        print(f"Error: {frontend_dir} directory not found")
        return False
    
    # Install dependencies
    if not run_command("npm install", cwd=frontend_dir):
        return False
    
    # Build the project
    if not run_command("npm run build", cwd=frontend_dir):
        print("Warning: Build failed, but development should still work")
    
    print("✓ Frontend setup complete")
    return True

def create_env_files():
    """Create environment files."""
    print("\n=== Creating Environment Files ===")
    
    # Backend .env
    backend_env = """# VocalScan Backend Configuration
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
PORT=8080
"""
    
    with open("backend/.env", "w") as f:
        f.write(backend_env)
    print("✓ Created backend/.env")
    
    # Frontend .env.local
    frontend_env = """# VocalScan Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8080
"""
    
    with open("frontend/.env.local", "w") as f:
        f.write(frontend_env)
    print("✓ Created frontend/.env.local")

def main():
    print("🎤 VocalScan Setup Script")
    print("This will set up the development environment for VocalScan.")
    
    # Check requirements
    print("\n=== Checking Requirements ===")
    
    # Check Python
    try:
        python_version = subprocess.check_output([sys.executable, "--version"], text=True).strip()
        print(f"✓ {python_version}")
    except:
        print("✗ Python not found")
        return False
    
    # Check Node.js
    try:
        node_version = subprocess.check_output(["node", "--version"], text=True).strip()
        print(f"✓ Node.js {node_version}")
    except:
        print("✗ Node.js not found. Please install Node.js 18+ from https://nodejs.org/")
        return False
    
    # Check npm
    try:
        npm_version = subprocess.check_output(["npm", "--version"], text=True).strip()
        print(f"✓ npm {npm_version}")
    except:
        print("✗ npm not found")
        return False
    
    # Create environment files
    create_env_files()
    
    # Setup backend
    if not setup_backend():
        print("❌ Backend setup failed")
        return False
    
    # Setup frontend
    if not setup_frontend():
        print("❌ Frontend setup failed")
        return False
    
    # Success message
    print("\n🎉 Setup Complete!")
    print("\nTo start the application:")
    print("\n1. Start the backend:")
    if platform.system() == "Windows":
        print("   cd backend")
        print("   venv\\Scripts\\activate")
        print("   python app/main.py")
    else:
        print("   cd backend")
        print("   source venv/bin/activate")
        print("   python app/main.py")
    
    print("\n2. Start the frontend (in a new terminal):")
    print("   cd frontend")
    print("   npm run dev")
    
    print("\n3. Open http://localhost:3000 in your browser")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
