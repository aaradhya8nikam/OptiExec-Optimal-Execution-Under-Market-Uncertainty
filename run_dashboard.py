"""
Launcher script to run the Streamlit Optimal Execution Engine dashboard.
Usage: python run_dashboard.py
"""

import os
import subprocess
import sys


def main():
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path, "--server.headless=true"]
    print(f"Launching Optimal Execution Engine Dashboard...")
    print(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
