"""
Security System — Entry Point

Run this file to launch the dashboard:
    python main.py

The app opens the camera, performs real-time face recognition, and shows the
live dashboard.  All configuration is in core/config.py.
"""

import logging
import os
import sys

# Ensure imports resolve from the project root regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    datefmt='%H:%M:%S',
)

from ui.app import SecurityApp


def main():
    app = SecurityApp()
    app.run()


if __name__ == '__main__':
    main()
