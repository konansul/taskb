#!/usr/bin/env python3
"""
Streamlit app runner with enhanced configuration
"""

import streamlit.web.cli as stcli
import sys
import os
from pathlib import Path

def run_app():
    """Run the Streamlit application with custom configuration"""
    # Set the working directory to the script's directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Set up sys.argv for streamlit run
    sys.argv = [
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    
    # Run the streamlit CLI
    stcli.main()

if __name__ == "__main__":
    run_app()
