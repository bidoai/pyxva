"""Thin launcher — kept for backwards compatibility.

Preferred usage:
    uv run risk-analytics-demo
    uv run python -m risk_analytics.demo
"""
from risk_analytics.demo import main

if __name__ == "__main__":
    main()
