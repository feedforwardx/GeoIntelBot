"""
File utility functions for reading and writing JSONL files.
"""
import json
from hashlib import md5

def write_jsonl_line(filepath, data):
    """Append a single JSON object as a line to a JSONL file."""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data) + '\n')

def read_jsonl_lines(filepath):
    """Yield each JSON object from a JSONL file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)
            
def encode_md5(text):
    return md5(text.encode("utf-8")).hexdigest()

