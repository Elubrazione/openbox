#!/usr/bin/env python3
"""
Build script for OpenBox visualization frontend.

This script builds the React frontend and copies the output to the static directory.
Run this script to update the static files after modifying the frontend.

Usage:
    python -m openbox.visualization.advance.build_frontend
    
Or from the project root:
    python openbox/visualization/advance/build_frontend.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def build_frontend():
    """Build the frontend and copy to static directory."""
    
    # Get paths
    viz_dir = Path(__file__).parent
    frontend_dir = viz_dir / 'frontend'
    static_dir = viz_dir / 'static'
    
    print("=" * 60)
    print("Building OpenBox Visualization Frontend")
    print("=" * 60)
    
    # Check if frontend directory exists
    if not frontend_dir.exists():
        print(f"Error: Frontend directory not found at {frontend_dir}")
        print("Please ensure the frontend source code is in openbox/visualization/advance/frontend/")
        sys.exit(1)
    
    # Check if npm is installed
    try:
        subprocess.run(['npm', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: npm is not installed. Please install Node.js and npm first.")
        sys.exit(1)
    
    # Change to frontend directory
    os.chdir(frontend_dir)
    
    # Install dependencies if needed
    if not (frontend_dir / 'node_modules').exists():
        print("\n📦 Installing dependencies...")
        result = subprocess.run(['npm', 'install'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error installing dependencies:\n{result.stderr}")
            sys.exit(1)
        print("   Dependencies installed successfully!")
    
    # Build the frontend
    print("\n🔨 Building frontend...")
    result = subprocess.run(['npm', 'run', 'build'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error building frontend:\n{result.stderr}")
        sys.exit(1)
    
    # Check if dist directory was created
    dist_dir = frontend_dir / 'dist'
    if not dist_dir.exists():
        print(f"Error: Build output not found at {dist_dir}")
        sys.exit(1)
    
    # Clear and copy to static directory
    print("\n📁 Copying build output to static directory...")
    if static_dir.exists():
        shutil.rmtree(static_dir)
    
    shutil.copytree(dist_dir, static_dir)
    
    # List the copied files
    print("\n✅ Build complete! Static files:")
    for f in sorted(static_dir.rglob('*')):
        if f.is_file():
            size = f.stat().st_size
            rel_path = f.relative_to(static_dir)
            print(f"   {rel_path} ({size:,} bytes)")
    
    print("\n" + "=" * 60)
    print("Frontend build successful!")
    print(f"Static files are in: {static_dir}")
    print("=" * 60)


if __name__ == '__main__':
    build_frontend()
