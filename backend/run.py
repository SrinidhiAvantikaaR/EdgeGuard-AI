#!/usr/bin/env python3
"""
EdgeGuard AI - Backend Server
Run with: python run.py
"""

import uvicorn
import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description="EdgeGuard AI Backend Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--train", action="store_true", help="Train model on startup")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    logging.info("Starting EdgeGuard AI Backend...")
    logging.info(f"AMD Optimized: Yes (using multi-threading)")
    logging.info(f"CPU Cores: {os.cpu_count()}")
    
    # Train model if requested
    if args.train:
        from models.train import train_model
        train_model()
    
    # Start server
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()