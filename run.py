#!/usr/bin/env python3
"""
run.py — Entry point for VitalWatch
Usage:  python3 run.py
        PORT=9000 python3 run.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
from server import main

if __name__ == "__main__":
    main(int(os.environ.get("PORT", 8000)))
