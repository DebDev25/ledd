#!/usr/bin/env python
"""Build the 224 crop archive from raw GenImage. See ledd/data/prepare.py for the
crop-not-resize and lossless-PNG rationale."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ledd.data.prepare import main

if __name__ == "__main__":
    main()
