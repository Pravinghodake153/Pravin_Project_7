"""
============================================================
server.py — Localhost Web Server for Project 7
============================================================
PURPOSE:
    Runs the RAG Web Dashboard and API locally on your computer.

HOW TO RUN:
    python3 server.py

THEN OPEN IN YOUR BROWSER:
    http://localhost:8000
============================================================
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import sys

# Add api directory to import path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from api.index import handler as ApiHandler


class LocalDevHandler(ApiHandler):
    """
    Combines static file serving from public/ with
    the serverless API handler from api/index.py.
    """

    def do_GET(self):
        # Route API requests to the API handler
        if self.path.startswith("/api/"):
            return super().do_GET()

        # Route web page requests to public/ directory
        clean_path = self.path.split("?")[0]
        if clean_path == "/" or clean_path == "":
            file_path = os.path.join(PROJECT_ROOT, "public", "index.html")
        else:
            file_path = os.path.join(PROJECT_ROOT, "public", clean_path.lstrip("/"))

        if os.path.exists(file_path) and os.path.isfile(file_path):
            self.send_response(200)
            if file_path.endswith(".html"):
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif file_path.endswith(".css"):
                self.send_header("Content-Type", "text/css; charset=utf-8")
            elif file_path.endswith(".js"):
                self.send_header("Content-Type", "application/javascript; charset=utf-8")
            elif file_path.endswith(".svg"):
                self.send_header("Content-Type", "image/svg+xml")
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            # Fallback to index.html
            index_path = os.path.join(PROJECT_ROOT, "public", "index.html")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(index_path, "rb") as f:
                self.wfile.write(f.read())


def run_local_server(port=8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, LocalDevHandler)
    print("=" * 65)
    print("  PROJECT 7: LOCAL DEVELOPMENT SERVER")
    print("=" * 65)
    print(f"  Server running locally at: http://localhost:{port}")
    print(f"  Web Dashboard:             http://localhost:{port}/")
    print(f"  API Endpoint:              http://localhost:{port}/api/answer?q=...")
    print("=" * 65)
    print("  Press Ctrl+C to stop the server.")
    print()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_local_server(port)
