"""
Logging utility functions for consistent info output.
"""
from datetime import datetime

def log_info(message):
    """Print an info message with a timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
