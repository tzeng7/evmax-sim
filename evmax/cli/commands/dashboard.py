"""evmax dashboard — launch the FastAPI web visualization."""
from __future__ import annotations

import subprocess
from pathlib import Path

import typer

app = typer.Typer(help="Launch the evmax web dashboard.")

ROOT = Path(__file__).resolve().parents[3]
FRONTEND = ROOT / "frontend"
DIST = FRONTEND / "dist"


def _maybe_build_spa(rebuild: bool) -> None:
    """Build the React SPA if dist/ is missing or --rebuild was passed."""
    if not FRONTEND.exists():
        return
    if rebuild or not (DIST / "index.html").exists():
        if not (FRONTEND / "node_modules").exists():
            typer.echo("Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=str(FRONTEND), check=True)
        typer.echo("Building frontend...")
        subprocess.run(["npm", "run", "build"], cwd=str(FRONTEND), check=True)


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(True, "--reload/--no-reload", help="Auto-reload on code changes."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild the React SPA before starting."),
) -> None:
    """Start the dashboard at http://HOST:PORT/.

    Serves the React SPA (with Portfolios) at /. The legacy Jinja dashboard
    is still available at /legacy. If frontend/dist is missing it is built
    automatically; pass --rebuild to force a rebuild after frontend edits.
    """
    import uvicorn

    _maybe_build_spa(rebuild)
    typer.echo(f"evmax dashboard → http://{host}:{port}/")
    uvicorn.run("evmax.web.app:app", host=host, port=port, reload=reload)
