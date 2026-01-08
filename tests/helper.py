import sys
import os

def set_wd():
    """Ensure project root is in sys.path for direct execution"""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))