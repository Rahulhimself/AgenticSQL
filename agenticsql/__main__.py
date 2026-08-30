"""
Module execution entry point for `python -m agenticsql`.
Delegates directly to the AgenticSQL CLI handler.
"""

from .cli import main

if __name__ == "__main__":
    # Run the interactive CLI when package is invoked with -m
    main()

