#!/usr/bin/env python3
"""Initialize the SQLite database schema.

Run once before first use:
    python scripts/init_db.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evmax.db import init_db
from evmax.settings import get_settings


async def main() -> None:
    settings = get_settings()
    print(f"Initializing database: {settings.database_url}")

    await init_db()

    print("Database schema created successfully.")
    print("\nNext steps:")
    print("  1. cp .env.example .env")
    print("  2. Edit .env with your API keys")
    print("  3. evmax scan --sectors NFL --once")


if __name__ == "__main__":
    asyncio.run(main())
