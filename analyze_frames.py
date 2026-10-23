#!/usr/bin/env python
"""Frame quality diagnosis before 3D reconstruction: `python analyze_frames.py --input DIR --output DIR`."""
import sys

from frame_analyzer.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
