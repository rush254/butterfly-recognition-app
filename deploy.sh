#!/bin/bash

# Create virtual environment
python -m venv antenv
source antenv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start the application using gunicorn
gunicorn --chdir /home/site/wwwroot app:app
