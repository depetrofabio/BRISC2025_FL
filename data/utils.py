### File for utils functions 
import os
import glob
import pandas as pd

# Function to get all file paths matching a pattern
def get_file_paths(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return files

def count_patterns(patterns):
    """
    Count the total number of files matching one or more glob patterns.

    Parameters
    ----------
    patterns : list of str
        List of glob-style file path patterns (e.g., ['path/to/files/*.jpg']).

    Returns
    -------
    int
        Total number of files matching the provided patterns.
    """
    total_count = 0
    for pat in patterns:
        matches = glob.glob(pat)
        total_count += len(matches)
    return total_count