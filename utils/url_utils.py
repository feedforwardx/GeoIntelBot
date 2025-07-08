"""
URL utility functions for normalization and type checking.
"""
from urllib.parse import urldefrag

def normalize_url(url):
    """Remove URL fragments for consistency."""
    return urldefrag(url)[0]

def is_html_page(url):
    """Return True if the URL is likely an HTML page, not a binary file."""
    non_html_exts = [".zip", ".tar", ".gz", ".rar", ".jar", ".exe", ".iso", ".7z", ".bz2"]
    return url.startswith("http") and not any(url.lower().endswith(ext) for ext in non_html_exts) 