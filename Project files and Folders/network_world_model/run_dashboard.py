"""
Run Dashboard - Launches the NetSentinel.AI Enterprise Threat Forecasting Platform.
Run:
    python run_dashboard.py
"""
import uvicorn
import webbrowser
import threading
import time
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# Student Project Note:
# Helper function that waits 1.5 seconds for the backend Uvicorn server to start,
# then automatically pops open the default web browser to the dashboard URL.
def open_browser():
    time.sleep(1.5)
    url = "http://localhost:5000"
    print(f"[LAUNCH] Opening NetSentinel.AI Dashboard in browser at {url}...")
    try:
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    print("=" * 75)
    print("  NETSENTINEL.AI - PREDICTIVE THREAT FORECASTING PLATFORM")
    print("  Powered by RSSM Latent World Models & Explainable AI (XAI)")
    print("=" * 75)
    print("[*] Starting backend server on http://localhost:5000...")
    
    # Step 1: Start background thread to open browser once server is live
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Step 2: Launch Uvicorn ASGI server hosting the FastAPI backend
    uvicorn.run("web_dashboard.server:app", host="0.0.0.0", port=5000, reload=False)
