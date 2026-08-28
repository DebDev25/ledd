#!/usr/bin/env python
"""RUN BEFORE ANY TRAINING. Exits non-zero if the preprocessing pipeline leaks
generator identity through the spectrum."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ledd.data.leak_check import main

if __name__ == "__main__":
    main()
