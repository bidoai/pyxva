"""Thin launcher — kept for backwards compatibility.

Preferred usage:
    uv run risk-analytics-demo
    uv run python -m pyxva.demo
"""
from pyxva.demo import main

if __name__ == "__main__":
    main()
