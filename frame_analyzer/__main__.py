"""Allows `python -m frame_analyzer --input DIR --output DIR`."""
import sys

from frame_analyzer.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
