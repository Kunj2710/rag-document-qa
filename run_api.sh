#!/bin/bash
# Script to run FastAPI server

echo "Starting RAG System API Server..."
echo "Make sure you have set up your .env file with API keys"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run API server
python api_server.py

