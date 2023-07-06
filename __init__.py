import os
import pathlib

def root_directory() -> pathlib.Path:
    """Returns the project's root directory"""
    return pathlib.Path(__file__).parent.parent
