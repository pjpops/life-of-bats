#!/bin/bash
# Double-click to launch Life of Bats, or drag this file to your Dock.
cd "$(dirname "$0")"
source venv/bin/activate
python3 bat_sonar.py
