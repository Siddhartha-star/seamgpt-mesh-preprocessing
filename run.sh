#!/usr/bin/env bash
set -e
echo "Installing requirements..."
pip install -r requirements.txt
echo "Running main pipeline..."
python main.py
echo "Generating PDF report..."
python create_report.py
echo "Done. Check output/ and mesh_assignment_report.pdf"
