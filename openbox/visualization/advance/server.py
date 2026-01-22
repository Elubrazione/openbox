"""
OpenBox Advanced Visualization Server

Provides a local HTTP server for advanced visualization features.
Based on Flask, serves the React frontend and provides API endpoints.
"""

import json
import math
import os
import socket
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from openbox import logger


def sanitize_json_values(obj: Any) -> Any:
    """
    Recursively sanitize JSON values to handle Infinity, -Infinity, and NaN.
    These are not valid JSON values and will cause parsing errors.
    """
    if isinstance(obj, dict):
        return {k: sanitize_json_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json_values(item) for item in obj]
    elif isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        elif math.isnan(obj):
            return None
        return obj
    return obj

try:
    from flask import Flask, jsonify, send_from_directory, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask not installed. Advanced visualization requires: pip install flask flask-cors")


class VisualizationServer:
    """
    Local HTTP server for OpenBox visualization.
    
    This server provides:
    - Static file serving for the React frontend
    - API endpoints for compression history data
    - Real-time data refresh support
    
    Example:
        >>> server = VisualizationServer(data_dir='./results/compression')
        >>> url = server.start(open_browser=True)
        >>> # Server runs in background, browser opens automatically
    """
    
    def __init__(
        self,
        data_dir: str,
        port: int = 8050,
        host: str = '127.0.0.1'
    ):
        """
        Initialize the visualization server.
        
        Args:
            data_dir: Directory containing compression_history.json
            port: Server port (will find available port if occupied)
            host: Server host address
        """
        if not FLASK_AVAILABLE:
            raise ImportError(
                "Flask is required for advanced visualization. "
                "Install with: pip install flask flask-cors"
            )
        
        self.data_dir = Path(data_dir)
        self.port = port
        self.host = host
        self._server_thread: Optional[threading.Thread] = None
        self._is_running = False
        
        # Get the static files directory (frontend build)
        self.static_dir = Path(__file__).parent / 'static'
        
        # Create Flask app
        self.app = self._create_app()
    
    def _create_app(self):
        """Create and configure the Flask application."""
        app = Flask(
            __name__,
            static_folder=str(self.static_dir),
            static_url_path=''
        )
        
        # Enable CORS for all routes
        CORS(app, resources={
            r"/*": {
                "origins": "*",
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type"],
            }
        })
        
        # Disable Flask default logging in production
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.WARNING)
        
        # Routes
        @app.route('/')
        def index():
            """Serve the main HTML page."""
            index_path = self.static_dir / 'index.html'
            if index_path.exists():
                return send_from_directory(str(self.static_dir), 'index.html')
            else:
                # Fallback: return embedded HTML
                return self._get_fallback_html()
        
        @app.route('/api/compression/history')
        def get_compression_history():
            """Get the latest compression history data."""
            try:
                history_file = self.data_dir / 'compression_history.json'
                
                if not history_file.exists():
                    # Try to find in subdirectories
                    for subdir in sorted(self.data_dir.glob("*"), 
                                        key=lambda p: p.stat().st_mtime if p.is_dir() else 0,
                                        reverse=True):
                        if subdir.is_dir():
                            candidate = subdir / 'compression_history.json'
                            if candidate.exists():
                                history_file = candidate
                                break
                
                if not history_file.exists():
                    return jsonify({
                        'success': False,
                        'error': 'No compression history found'
                    }), 404
                
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                
                # Wrap in object if it's a list (for backward compatibility)
                if isinstance(history_data, list):
                    history_data = {'history': history_data}
                
                # Add metadata
                history_data['output_dir'] = str(self.data_dir)
                history_data['created_at'] = datetime.fromtimestamp(
                    history_file.stat().st_ctime
                ).isoformat()
                history_data['last_modified'] = datetime.fromtimestamp(
                    history_file.stat().st_mtime
                ).isoformat()
                
                # Sanitize data to handle Infinity/NaN values
                sanitized_data = sanitize_json_values(history_data)
                
                return jsonify({
                    'success': True,
                    'data': sanitized_data
                })
            
            except Exception as e:
                logger.error(f"Error loading compression history: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'service': 'OpenBox Visualization Server',
                'data_dir': str(self.data_dir),
                'timestamp': datetime.now().isoformat()
            })
        
        @app.route('/api/upload', methods=['POST'])
        def upload_files():
            """
            Upload compression history JSON file.
            
            Accepts:
            - history: One or more history JSON files (multipart form data)
            - Or JSON body with compression_history data directly
            
            Returns the uploaded/processed data for visualization.
            """
            try:
                # Check if it's a file upload or JSON body
                if request.files:
                    # Handle file upload
                    history_files = request.files.getlist('history')
                    if not history_files:
                        return jsonify({
                            'success': False,
                            'error': 'No history files provided'
                        }), 400
                    
                    # Process the first history file
                    history_file = history_files[0]
                    if history_file.filename == '':
                        return jsonify({
                            'success': False,
                            'error': 'Empty filename'
                        }), 400
                    
                    # Read and validate JSON
                    try:
                        file_content = history_file.read().decode('utf-8')
                        history_data = json.loads(file_content)
                    except json.JSONDecodeError as e:
                        return jsonify({
                            'success': False,
                            'error': f'Invalid JSON: {str(e)}'
                        }), 400
                    
                    # Save to data directory
                    output_path = self.data_dir / 'compression_history.json'
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(history_data, f, indent=2)
                    
                    # Sanitize data for JSON response
                    sanitized_data = sanitize_json_values(history_data)
                    
                    return jsonify({
                        'success': True,
                        'message': 'History file uploaded successfully',
                        'data': sanitized_data
                    })
                
                elif request.is_json:
                    # Handle JSON body directly
                    history_data = request.get_json()
                    
                    if not history_data:
                        return jsonify({
                            'success': False,
                            'error': 'Empty JSON body'
                        }), 400
                    
                    # Save to data directory
                    output_path = self.data_dir / 'compression_history.json'
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(history_data, f, indent=2)
                    
                    # Sanitize data for JSON response
                    sanitized_data = sanitize_json_values(history_data)
                    
                    return jsonify({
                        'success': True,
                        'message': 'History data saved successfully',
                        'data': sanitized_data
                    })
                
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No data provided. Send files via multipart/form-data or JSON body.'
                    }), 400
                    
            except Exception as e:
                logger.error(f"Error uploading files: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        # Serve static files (JS, CSS, etc.)
        @app.route('/<path:path>')
        def serve_static(path):
            """Serve static files."""
            # Don't handle API routes here
            if path.startswith('api/'):
                return jsonify({'success': False, 'error': 'API endpoint not found'}), 404
            
            file_path = self.static_dir / path
            if file_path.exists():
                return send_from_directory(str(self.static_dir), path)
            # Fallback to index.html for SPA routing
            return send_from_directory(str(self.static_dir), 'index.html')
        
        return app
    
    def _get_fallback_html(self) -> str:
        """Return fallback HTML when static files are not available."""
        return '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OpenBox Visualization</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container {
            text-align: center;
            padding: 40px;
            background: rgba(255,255,255,0.1);
            border-radius: 16px;
            backdrop-filter: blur(10px);
        }
        h1 { margin-bottom: 20px; }
        p { opacity: 0.9; margin: 10px 0; }
        code {
            background: rgba(0,0,0,0.2);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 14px;
        }
        .status { margin-top: 30px; }
        .api-link {
            color: #a5d6ff;
            text-decoration: none;
        }
        .api-link:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 OpenBox Visualization Server</h1>
        <p>Server is running, but frontend files are not found.</p>
        <p>To build the frontend, run:</p>
        <p><code>cd openbox/visualization/advance/frontend && npm install && npm run build</code></p>
        <div class="status">
            <p>API Status: <a href="/api/health" class="api-link">/api/health</a></p>
            <p>Compression History: <a href="/api/compression/history" class="api-link">/api/compression/history</a></p>
        </div>
    </div>
</body>
</html>
'''
    
    def _find_available_port(self, start_port: int) -> int:
        """Find an available port starting from start_port."""
        port = start_port
        max_attempts = 100
        
        for _ in range(max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((self.host, port))
                    return port
            except OSError:
                port += 1
        
        raise RuntimeError(f"Could not find available port after {max_attempts} attempts")
    
    def start(self, open_browser: bool = True, blocking: bool = False) -> str:
        """
        Start the visualization server.
        
        Args:
            open_browser: Whether to open the browser automatically
            blocking: If True, block until server is stopped. If False, run in background.
        
        Returns:
            The URL where the server is running
        """
        # Find available port
        try:
            self.port = self._find_available_port(self.port)
        except RuntimeError as e:
            logger.error(str(e))
            raise
        
        url = f'http://{self.host}:{self.port}'
        
        # Print startup banner
        self._print_banner(url)
        
        def safe_open_browser(url):
            """Safely open browser, handling headless environments."""
            try:
                webbrowser.open(url)
            except Exception as e:
                logger.warning(f"Could not open browser automatically: {e}")
                logger.info(f"Please open manually: {url}")
        
        if blocking:
            # Run in main thread (blocking)
            if open_browser:
                # Open browser after a short delay
                threading.Timer(1.0, lambda: safe_open_browser(url)).start()
            
            self._is_running = True
            self.app.run(
                host=self.host,
                port=self.port,
                threaded=True,
                use_reloader=False
            )
        else:
            # Run in background thread
            def run_server():
                self._is_running = True
                self.app.run(
                    host=self.host,
                    port=self.port,
                    threaded=True,
                    use_reloader=False
                )
            
            self._server_thread = threading.Thread(target=run_server, daemon=True)
            self._server_thread.start()
            
            # Wait a moment for server to start
            import time
            time.sleep(0.5)
            
            if open_browser:
                safe_open_browser(url)
        
        return url
    
    def _print_banner(self, url: str):
        """Print server startup banner."""
        print("=" * 72)
        print(" " * 16 + "OpenBox Visualization Server")
        print("=" * 72)
        print(f"\n📊 Data directory: {self.data_dir.absolute()}")
        print(f"\n🚀 Server running at: {url}")
        print(f"   API endpoint: {url}/api/compression/history")
        
        if self.host == '0.0.0.0':
            # Get local IP for remote access
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                print(f"   Remote access: http://{local_ip}:{self.port}")
            except Exception:
                pass
        
        print(f"\n   Press Ctrl+C to stop the server")
        print("=" * 72 + "\n")
    
    def stop(self):
        """Stop the visualization server."""
        self._is_running = False
        # Note: Flask doesn't have a clean way to stop from another thread
        # The daemon thread will stop when the main program exits
        logger.info("Visualization server stopping...")
    
    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._is_running


def start_visualization_server(
    data_dir: str,
    port: int = 8050,
    host: str = '127.0.0.1',
    open_browser: bool = True,
    blocking: bool = False
) -> str:
    """
    Start the OpenBox visualization server.
    
    This is a convenience function that creates a VisualizationServer
    and starts it.
    
    Args:
        data_dir: Directory containing compression_history.json
        port: Server port (default: 8050)
        host: Server host (default: 127.0.0.1, use 0.0.0.0 for remote access)
        open_browser: Whether to open browser automatically
        blocking: If True, block until Ctrl+C. If False, run in background.
    
    Returns:
        The URL where the server is running
    
    Example:
        >>> url = start_visualization_server('./results/compression')
        >>> print(f"Server running at {url}")
    """
    server = VisualizationServer(data_dir=data_dir, port=port, host=host)
    return server.start(open_browser=open_browser, blocking=blocking)


def visualize_compression(
    data_dir: str,
    mode: str = 'advanced',
    port: int = 8050,
    open_browser: bool = True
) -> str:
    """
    Visualize compression results.
    
    This is the main entry point for standalone visualization.
    
    Args:
        data_dir: Directory containing compression_history.json
        mode: 'basic' for static HTML, 'advanced' for local server
        port: Server port (only for advanced mode)
        open_browser: Whether to open browser automatically
    
    Returns:
        Path to HTML file (basic mode) or server URL (advanced mode)
    
    Example:
        >>> from openbox.visualization.advance import visualize_compression
        >>> 
        >>> # Advanced mode (local server)
        >>> url = visualize_compression('./results', mode='advanced')
        >>> 
        >>> # Basic mode (static HTML)
        >>> html_path = visualize_compression('./results', mode='basic')
    """
    if mode == 'basic':
        from .html_generator import generate_static_html
        return generate_static_html(
            data_dir=data_dir,
            open_browser=open_browser
        )
    else:
        return start_visualization_server(
            data_dir=data_dir,
            port=port,
            open_browser=open_browser,
            blocking=False
        )
