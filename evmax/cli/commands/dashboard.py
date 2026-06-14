"""evmax dashboard — launch the FastAPI web visualization."""
from __future__ import annotations

import subprocess
from pathlib import Path

import typer

app = typer.Typer(help="Launch the evmax web dashboard.")

ROOT = Path(__file__).resolve().parents[3]
FRONTEND = ROOT / "frontend"
DIST = FRONTEND / "dist"

# Top-level frontend files (outside src/) that affect the build output. A change
# to any of these should also trigger a rebuild.
_SOURCE_ROOT_FILES = (
    "index.html",
    "package.json",
    "vite.config.ts",
    "tsconfig.json",
    "tsconfig.app.json",
    "tsconfig.node.json",
)


def _latest_source_mtime() -> float:
    """Newest mtime across the SPA source tree (frontend/src + key config files).

    node_modules and dist are excluded — only files that feed the build count.
    Returns 0.0 if no source is found (treated as "nothing to build").
    """
    latest = 0.0
    src = FRONTEND / "src"
    if src.exists():
        for path in src.rglob("*"):
            if path.is_file():
                latest = max(latest, path.stat().st_mtime)
    for name in _SOURCE_ROOT_FILES:
        f = FRONTEND / name
        if f.exists():
            latest = max(latest, f.stat().st_mtime)
    return latest


def _spa_is_stale() -> bool:
    """True when the built dist/ is older than the newest SPA source file.

    This is what prevents serving a stale UI: if you edit the frontend and
    forget to rebuild, the next `serve` rebuilds automatically.
    """
    index = DIST / "index.html"
    if not index.exists():
        return True
    return _latest_source_mtime() > index.stat().st_mtime


def _maybe_build_spa(rebuild: bool) -> None:
    """Build the React SPA if dist/ is missing, stale, or --rebuild was passed."""
    if not FRONTEND.exists():
        return
    if rebuild or _spa_is_stale():
        if not (FRONTEND / "node_modules").exists():
            typer.echo("Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=str(FRONTEND), check=True)
        reason = "forced" if rebuild else "source changed since last build"
        typer.echo(f"Building frontend ({reason})...")
        subprocess.run(["npm", "run", "build"], cwd=str(FRONTEND), check=True)


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(True, "--reload/--no-reload", help="Auto-reload on code changes."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Force a rebuild of the React SPA before starting."),
    build: bool = typer.Option(True, "--build/--no-build", help="Auto-build the SPA when dist/ is missing or stale. Use --no-build for a fast start."),
) -> None:
    """Start the dashboard at http://HOST:PORT/.

    Serves the React SPA (with Portfolios) at /. The SPA is built automatically
    whenever frontend/dist is missing OR the frontend source is newer than the
    last build, so you never serve a stale UI. Pass --rebuild to force it, or
    --no-build to skip the check entirely for a faster start.
    """
    import uvicorn

    if build or rebuild:  # --rebuild forces a build even with --no-build
        _maybe_build_spa(rebuild)
    typer.echo(f"evmax dashboard → http://{host}:{port}/")
    uvicorn.run("evmax.web.app:app", host=host, port=port, reload=reload)
