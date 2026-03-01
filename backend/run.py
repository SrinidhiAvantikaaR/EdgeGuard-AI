#!/usr/bin/env python3
"""
EdgeGuard AI - Unified Launcher
Starts both backend server and serves frontend
"""

import os
import sys
import subprocess
import webbrowser
import time
import threading
import socket
import platform

def is_port_in_use(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def open_browser():
    """Open browser after a short delay"""
    time.sleep(3)
    webbrowser.open('http://localhost:8000/app/')
    print("\n🌐 Browser opened to http://localhost:8000/app/")

def main():
    print("=" * 60)
    print("🚀 EdgeGuard AI - Real-Time Ransomware Detection")
    print("=" * 60)
    
    # Check if port is available
    if is_port_in_use(8000):
        print("❌ Port 8000 is already in use!")
        response = input("Do you want to kill the process using port 8000? (y/n): ")
        if response.lower() == 'y':
            if platform.system() == 'Windows':
                subprocess.run(['netstat', '-ano', '|', 'findstr', ':8000'], shell=True)
                # You'd need to parse and kill manually on Windows
            else:
                subprocess.run(['fuser', '-k', '8000/tcp'])
            time.sleep(2)
        else:
            print("Please free port 8000 and try again.")
            sys.exit(1)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = script_dir  # Since run.py is in backend
    
    # Path to main.py
    main_py = os.path.join(backend_dir, "main.py")
    
    if not os.path.exists(main_py):
        print(f"❌ Could not find main.py at {main_py}")
        print("Please ensure run.py is in the same directory as main.py")
        sys.exit(1)
    
    # Check if frontend exists
    frontend_dir = os.path.join(os.path.dirname(backend_dir), "frontend")
    index_html = os.path.join(frontend_dir, "index.html")
    
    if not os.path.exists(index_html):
        print(f"⚠️  Frontend not found at {index_html}")
        print("Creating frontend directory...")
        os.makedirs(frontend_dir, exist_ok=True)
        
        # Ask user to place the HTML file
        print("\n📋 Please save your HTML frontend as:")
        print(f"   {index_html}")
        print("\nThen run this script again.")
        sys.exit(1)
    
    print(f"✅ Found backend: {main_py}")
    print(f"✅ Found frontend: {index_html}")
    
    # Start browser thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    print("\n⚡ Starting EdgeGuard AI server...")
    print("   Press Ctrl+C to stop\n")
    
    try:
        # Run the FastAPI server
        os.chdir(backend_dir)  # Change to backend directory
        subprocess.run([
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down EdgeGuard AI...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()