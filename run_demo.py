"""
Master Demo Launcher for Sovereign Mathematical Optimization Engine.
Boots both the FastAPI REST engine and the Next.js interactive web frontend.
"""
import subprocess
import sys
import time
import os
import signal

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(root_dir, "frontend")

    print("=" * 70)
    print("  SOVEREIGN MATHEMATICAL OPTIMIZATION ENGINE - DEMO LAUNCHER")
    print("  AI-Guided • Sparse-First • Sovereign Mathematical Foundations")
    print("=" * 70)

    # 1. Start FastAPI Backend on port 8000
    print("\n[1/2] Launching FastAPI REST Engine on http://localhost:8000 ...")
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "sovereign_opt.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    backend_proc = subprocess.Popen(backend_cmd, cwd=root_dir)

    # Wait 2 seconds for backend to initialize
    time.sleep(2)

    # 2. Start Next.js Frontend on port 3000
    print("[2/2] Launching Next.js Interactive Dashboard on http://localhost:3000 ...")
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_cmd = [npm_cmd, "run", "dev"]
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=frontend_dir)

    print("\n" + "=" * 70)
    print("  SYSTEM READY!")
    print("  - Interactive Web Dashboard: http://localhost:3000")
    print("  - FastAPI Interactive Docs:   http://localhost:8000/docs")
    print("  - Rich Terminal CLI:         python -m sovereign_opt.cli.app demo")
    print("=" * 70)
    print("\nPress Ctrl+C at any time to shut down both servers.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down demo services...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Done. Have a great day!")


if __name__ == "__main__":
    main()
