"""Run with: streamlit run app.py"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent / "src"))
from ai_buckingham.ui.app import main

main()
