#!/bin/bash
# First-time dev setup. Run once after cloning.
set -e

pip install -e ".[dev]"
pre-commit install

echo ""
echo "Setup complete. Git hooks installed."
echo "Run 'pytest tests/ -q' to verify."
