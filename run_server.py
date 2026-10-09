"""
FastAPI Server Launcher for Optimal Execution Engine.
Starts the Uvicorn ASGI server and serves the web application and REST API at http://localhost:8000.

Usage:
    python run_server.py
"""

import os
import sys
import webbrowser
import threading
import time
import uvicorn


def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://localhost:8000")


def main():
    print("=" * 80)
    print(" ⚡ OPTIMAL EXECUTION ENGINE - FASTAPI WEB APPLICATION SERVER")
    print("=" * 80)
    print(" • Web App & Dashboard : http://localhost:8000")
    print(" • Interactive Swagger  : http://localhost:8000/docs")
    print(" • OpenAPI JSON Schema  : http://localhost:8000/openapi.json")
    print("=" * 80)

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
