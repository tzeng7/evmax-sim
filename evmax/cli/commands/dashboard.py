"""evmax dashboard — launch the FastAPI web visualization."""
from __future__ import annotations

import typer

app = typer.Typer(help="Launch the evmax web dashboard.")


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    """Start the dashboard at http://HOST:PORT/."""
    import uvicorn

    typer.echo(f"evmax dashboard → http://{host}:{port}/")
    uvicorn.run("evmax.web.app:app", host=host, port=port, reload=reload)
