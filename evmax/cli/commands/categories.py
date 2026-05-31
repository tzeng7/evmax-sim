"""CLI commands for inspecting and managing the betting-category registry.

Subcommands:
  evmax categories list               — print all registered categories as a table
  evmax categories show <key>         — detail view for one category
  evmax categories validate           — run validate_registry(), exit non-zero on error
  evmax categories modes              — show effective mode per category (runtime overrides applied)

The registry itself lives in ``data/categories.yaml`` and is loaded by
``evmax.categories``. These commands are read-only; edits happen by
editing the YAML directly or by running ``evmax cleanup shadow promote``
(which lives in the cleanup CLI family).
"""

from __future__ import annotations

import sys

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    help=(
        "Inspect the betting-category registry "
        "(data/categories.yaml — the source of truth for what evmax can bet on, "
        "which models contribute, and what mode each category runs in)."
    )
)
console = Console()


_MODE_COLOR = {
    "live": "green",
    "shadow": "yellow",
    "disabled": "red",
}

_STATUS_COLOR = {
    "shipped": "green",
    "wip": "yellow",
    "unresolved": "yellow",
    "blocked": "red",
}


def _color_mode(mode: str) -> str:
    color = _MODE_COLOR.get(mode, "white")
    return f"[{color}]{mode}[/{color}]"


def _color_status(status: str) -> str:
    color = _STATUS_COLOR.get(status, "white")
    return f"[{color}]{status}[/{color}]"


@app.command("list")
def list_categories(
    mode: str = typer.Option(
        None, "--mode", "-m",
        help="Filter by mode: live, shadow, disabled. Default: all.",
    ),
    status: str = typer.Option(
        None, "--status",
        help="Filter by status: shipped, wip, unresolved, blocked. Default: all.",
    ),
    effective: bool = typer.Option(
        False, "--effective",
        help="Show effective mode (runtime / env var overrides applied) "
             "instead of the YAML base mode.",
    ),
) -> None:
    """Print every registered category as a table.

    Columns: key, display_name, market_types, models, mode, resolver, status.
    """
    from evmax.categories import all_categories
    from evmax.modes import get_mode

    rows = all_categories()
    if mode:
        rows = [r for r in rows if r.mode == mode]
    if status:
        rows = [r for r in rows if r.status == status]

    if not rows:
        console.print("[yellow]No categories match the filter.[/yellow]")
        raise typer.Exit(0)

    title_suffix = " (effective mode)" if effective else ""
    table = Table(
        title=f"Betting categories{title_suffix}",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("Key", width=12, no_wrap=True)
    table.add_column("Display name", min_width=16, no_wrap=False)
    table.add_column("Market types", no_wrap=False)
    table.add_column("Models", no_wrap=False)
    table.add_column("Mode", width=10)
    table.add_column("Resolver", width=18)
    table.add_column("Status", width=12)

    for spec in rows:
        if effective:
            effective_mode = get_mode(spec.key)
            mode_cell = _color_mode(effective_mode)
            if effective_mode != spec.mode:
                mode_cell += f" [dim](yaml: {spec.mode})[/dim]"
        else:
            mode_cell = _color_mode(spec.mode)

        market_str = ", ".join(mt.value for mt in spec.market_types)
        models_str = ", ".join(spec.models)
        table.add_row(
            spec.key,
            spec.display_name,
            market_str,
            models_str,
            mode_cell,
            spec.resolver,
            _color_status(spec.status),
        )

    console.print(table)
    console.print(
        f"[dim]{len(rows)} categories. "
        f"Source of truth: data/categories.yaml[/dim]"
    )


@app.command("show")
def show_category(
    key: str = typer.Argument(..., help="Category key (e.g. 'nba', 'nfl_props')"),
) -> None:
    """Detail view for one category, including notes and prop stat types."""
    from evmax.categories import get_category
    from evmax.modes import get_mode

    try:
        spec = get_category(key)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    effective_mode = get_mode(spec.key)
    mode_display = _color_mode(spec.mode)
    if effective_mode != spec.mode:
        mode_display += f" [dim](currently overridden to [/dim]{_color_mode(effective_mode)}[dim])[/dim]"

    lines = [
        f"[bold]Key:[/bold]          {spec.key}",
        f"[bold]Display name:[/bold] {spec.display_name}",
        f"[bold]Mode:[/bold]         {mode_display}",
        f"[bold]Status:[/bold]       {_color_status(spec.status)}",
        f"[bold]Resolver:[/bold]     {spec.resolver}",
        "",
        f"[bold]Market types[/bold] ({len(spec.market_types)}):",
        "  " + ", ".join(
            f"{mt.value} [yellow](shadow)[/yellow]" if mt.value in spec.shadow_market_types
            else mt.value
            for mt in spec.market_types
        ),
    ]
    if spec.prop_stat_types:
        lines.append("")
        lines.append(f"[bold]Prop stat types[/bold] ({len(spec.prop_stat_types)}):")
        lines.append("  " + ", ".join(spec.prop_stat_types))
    lines.append("")
    lines.append(f"[bold]Models[/bold] ({len(spec.models)}):")
    lines.append("  " + ", ".join(spec.models))

    if spec.notes:
        lines.append("")
        lines.append(f"[bold]Notes:[/bold]")
        for para in spec.notes.split(". "):
            para = para.strip()
            if para:
                lines.append(f"  {para}{'' if para.endswith('.') else '.'}")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold cyan]{spec.display_name}[/bold cyan]",
            box=box.ROUNDED,
            border_style="cyan",
        )
    )


@app.command("validate")
def validate_cli() -> None:
    """Run validate_registry() and print the result. Exits non-zero on error.

    Intended for CI and pre-release checks. Re-loads the YAML from disk so
    you can test edits without restarting.
    """
    from evmax.categories import reload_registry, validate_registry

    try:
        reload_registry()
        validate_registry()
    except Exception as e:
        console.print(f"[red]❌ Registry invalid:[/red] {e}")
        raise typer.Exit(1)

    from evmax.categories import all_categories

    console.print(
        f"[green]✓ Registry valid.[/green] "
        f"{len(all_categories())} categories in data/categories.yaml."
    )


@app.command("modes")
def show_modes(
    show_overrides: bool = typer.Option(
        True, "--overrides/--base",
        help="Show effective modes (including runtime / env overrides) vs the YAML base.",
    ),
) -> None:
    """Show the current mode for every category.

    By default shows the effective mode — runtime overrides from
    set_runtime_overrides() and env-var overrides from
    EVMAX_CATEGORY_MODES are applied.

    Pass --base to see only the YAML base modes (what will persist after
    the process exits).
    """
    from evmax.categories import all_categories
    from evmax.modes import get_mode

    table = Table(
        title="Category modes",
        box=box.SIMPLE,
        header_style="bold cyan",
    )
    table.add_column("Key", width=14, no_wrap=True)
    table.add_column("Effective" if show_overrides else "YAML base", width=12)
    if show_overrides:
        table.add_column("YAML base", width=12)
        table.add_column("Overridden?", width=12)

    for spec in all_categories():
        if show_overrides:
            effective = get_mode(spec.key)
            overridden = "yes" if effective != spec.mode else ""
            table.add_row(
                spec.key,
                _color_mode(effective),
                _color_mode(spec.mode),
                overridden,
            )
        else:
            table.add_row(spec.key, _color_mode(spec.mode))

    console.print(table)
