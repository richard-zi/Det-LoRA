#!/usr/bin/env python3
"""Wrapper entrypoint for the final thesis suite runner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from det_lora.final_runner import main

if __name__ == "__main__":
    main()
