#!/bin/bash
# Script to run Streamlit application

echo "Starting RAG System Streamlit Application..."
echo "Make sure you have set up your .env file with API keys"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run Streamlit
streamlit run streamlit_app.py

