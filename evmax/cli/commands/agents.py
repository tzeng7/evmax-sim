"""CLI commands for the agent pipeline and model management.

Commands:
  evmax agents scan  — run the agent coordinator for a single cycle and print gaps
  evmax agents seed  — seed Elo / Poisson / Form models from a JSON file
  evmax agents ratings — display current Elo ratings for a sector
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("scan")
def scan(
    sectors: str = typer.Option(
        "nba,ncaab,soccer,lol,cs2",
        "--sectors",
        "-s",
        help="Comma-separated sector list, e.g. 'nba,soccer'",
    ),
    no_models: bool = typer.Option(False, "--no-models", help="Skip model agents (sharp probs only)."),
    no_injuries: bool = typer.Option(False, "--no-injuries", help="Skip injury report agent."),
    sharp_weight: float = typer.Option(0.85, "--sharp-weight", help="Weight for Pinnacle in ensemble blend."),
    bankroll: float = typer.Option(250.0, "--bankroll", "-b", help="Current bankroll in USD."),
    kelly: float = typer.Option(0.5, "--kelly", "-k", help="Kelly fraction (0.5=half, 0.25=quarter)."),
    min_ev: float = typer.Option(0.02, "--min-ev", help="Minimum EV threshold to display."),
    top: int = typer.Option(25, "--top", "-n", help="Max plays to show."),
    date_filter: Optional[str] = typer.Option(
        None, "--date", "-d",
        help="Only show games on this date (YYYY-MM-DD). Defaults to today.",
    ),
) -> None:
    """Run the full agent pipeline for one cycle and display +EV plays."""
    from evmax.agents.coordinator import AgentCoordinator

    sector_list = [s.strip() for s in sectors.split(",") if s.strip()]

    # Read persisted sharp_weight from model_config.json (overrides CLI default 0.85)
    from evmax.agents.cleanup.metrics import load_config as _load_cfg
    _cfg = _load_cfg()
    if _cfg.get("sharp_weight") and sharp_weight == 0.85:
        sharp_weight = _cfg["sharp_weight"]

    coordinator = AgentCoordinator(
        sectors=sector_list,
        enable_models=not no_models,
        sharp_weight=sharp_weight,
        enable_injuries=not no_injuries,
        bankroll=bankroll,
        kelly_fraction=kelly,
    )

    console.print(f"\n[bold cyan]evmax agent scan[/bold cyan] — sectors: {', '.join(sector_list)}\n")

    result = asyncio.run(coordinator.run_cycle())

    # Auto-log all +EV gaps found (above min_ev) to predictions.db
    gaps_to_log = [g for g in result.top_gaps if g.ev_pct >= min_ev]
    if gaps_to_log:
        try:
            from evmax.agents.cleanup.logger import log_gaps as _log_gaps
            n_logged = _log_gaps(gaps_to_log, sharp_weight_used=sharp_weight)
            if n_logged:
                console.print(f"[dim]  Logged {n_logged} new prediction(s) to predictions.db[/dim]")
        except Exception as _log_err:
            console.print(f"[dim yellow]  Warning: could not log predictions: {_log_err}[/dim yellow]")

    # Print injury summary first
    if result.injury_reports:
        inj_lines = []
        for sector, team_reports in result.injury_reports.items():
            sig = [r for r in team_reports.values() if r.has_significant_injuries]
            if sig:
                inj_lines.append(f"[bold]{sector.upper()}[/bold]: " + ", ".join(
                    f"{r.team} ({len(r.players)} out/dtd, adj={r.probability_adjustment:+.1%})"
                    for r in sig[:5]
                ))
        if inj_lines:
            console.print("\n[bold yellow]Injury Impact:[/bold yellow]")
            for line in inj_lines:
                console.print(f"  {line}")

    # Parse date filter (default: today)
    if date_filter:
        try:
            target_date = datetime.strptime(date_filter, "%Y-%m-%d").date()
        except ValueError:
            console.print(f"[red]Invalid date format:[/red] {date_filter!r} — use YYYY-MM-DD")
            raise typer.Exit(1)
    else:
        target_date = date.today()

    def _matches_date(g) -> bool:
        if g.event_date is None:
            return True  # no date info — include by default
        # datetime subclasses date, so always call .date() to strip time/tz
        ed = g.event_date.date() if hasattr(g.event_date, "date") else g.event_date
        return ed == target_date

    gaps = [g for g in result.top_gaps if g.ev_pct >= min_ev and _matches_date(g)][:top]

    if not gaps:
        console.print(f"\n[yellow]No +EV plays found at EV >= {min_ev*100:.0f}% threshold.[/yellow]")
        console.print(f"Scanned {result.markets_fetched} markets, matched {result.markets_matched}.")
        if result.errors:
            console.print(f"[red]Errors:[/red] {', '.join(result.errors)}")
        return

    table = Table(
        title=f"+EV Plays — {len(gaps)} found | {target_date} | Bankroll ${bankroll:.0f} | {kelly:.0%} Kelly",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Sector", style="dim", width=6)
    table.add_column("Event", style="dim", min_width=24)
    table.add_column("Outcome", style="bold white", min_width=18)
    table.add_column("Kalshi", justify="right", width=7)
    table.add_column("True P", justify="right", width=7)
    table.add_column("EV %", justify="right", style="green bold", width=7)
    table.add_column("Kelly%", justify="right", width=7)
    table.add_column("Stake $", justify="right", style="cyan bold", width=8)
    table.add_column("Vol $", justify="right", width=9)
    table.add_column("Sources", style="dim", min_width=12)

    total_stake = 0.0
    for i, gap in enumerate(gaps, 1):
        stake = result.stake_for(gap)
        total_stake += stake
        ev_color = "bold green" if gap.ev_pct >= 0.10 else "green" if gap.ev_pct >= 0.05 else "yellow"
        table.add_row(
            str(i),
            gap.sector.upper(),
            gap.event_title[:28],
            gap.display_label[:22],
            f"{gap.kalshi_yes_price:.2f}",
            f"{gap.blended_true_prob:.3f}",
            f"[{ev_color}]{gap.ev_pct*100:+.1f}%[/{ev_color}]",
            f"{gap.kelly_fraction*100:.2f}%",
            f"${stake:.2f}",
            f"${gap.volume_usd:,.0f}",
            gap.model_sources[:14],
        )

    console.print(f"\n[bold cyan]evmax agent scan — {', '.join(sector_list).upper()}[/bold cyan]\n")
    console.print(table)
    console.print(
        f"\n  [bold]Total at risk:[/bold] ${total_stake:.2f} / ${bankroll:.0f} "
        f"({total_stake/bankroll*100:.1f}%)  |  "
        f"Matched {result.markets_matched}/{result.markets_fetched} markets\n"
    )

    if result.errors:
        console.print(f"[red]Errors:[/red] {', '.join(result.errors)}")


@app.command("seed")
def seed(
    model: str = typer.Argument(..., help="Model to seed: elo | form | poisson"),
    sector: str = typer.Option(..., "--sector", "-s", help="Sector, e.g. 'nba'"),
    file: Path = typer.Option(..., "--file", "-f", help="JSON file with seed data"),
) -> None:
    """Seed a model agent with historical data from a JSON file.

    JSON format by model type:

    elo:
      {"ratings": {"lakers": 1550.0, "celtics": 1620.0, ...}}

    form:
      {"results": [
        {"date": "2026-01-15", "home": "lakers", "away": "celtics",
         "score_home": 112, "score_away": 108},
        ...
      ]}

    poisson:
      {
        "league_avg": {"home": 1.55, "away": 1.15},
        "teams": {
          "manchester city": {"attack": 1.42, "defense": 0.65, "games": 28},
          ...
        }
      }
    """
    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(1)

    data = json.loads(file.read_text())

    if model == "elo":
        from evmax.agents.models.elo_agent import EloModelAgent
        agent = EloModelAgent()
        ratings = data.get("ratings", data)  # allow top-level dict too
        agent.seed_ratings(sector, ratings)
        console.print(f"[green]Seeded Elo for {sector}:[/green] {len(ratings)} teams")

    elif model == "form":
        from evmax.agents.models.form_agent import FormModelAgent
        agent = FormModelAgent()
        results = data.get("results", data)
        agent.seed_results(sector, results)
        console.print(f"[green]Seeded Form for {sector}:[/green] {len(results)} results")

    elif model == "poisson":
        from evmax.agents.models.poisson_agent import PoissonModelAgent
        agent = PoissonModelAgent()
        agent.seed_team_stats(
            sector=sector,
            team_stats=data.get("teams", {}),
            league_avg=data.get("league_avg"),
        )
        console.print(f"[green]Seeded Poisson for {sector}:[/green] {len(data.get('teams', {}))} teams")

    else:
        console.print(f"[red]Unknown model:[/red] {model}. Choose: elo | form | poisson")
        raise typer.Exit(1)


@app.command("ratings")
def ratings(
    sector: str = typer.Argument(..., help="Sector, e.g. 'nba'"),
    top: int = typer.Option(20, "--top", "-n", help="Show top N teams by Elo."),
) -> None:
    """Display current Elo ratings for a sector."""
    from evmax.agents.models.elo_agent import EloModelAgent
    agent = EloModelAgent()
    all_r = agent.all_ratings(sector)

    if not all_r:
        console.print(f"[yellow]No Elo ratings found for {sector}.[/yellow] Use [bold]evmax agents seed elo[/bold] to load data.")
        return

    sorted_teams = sorted(all_r.items(), key=lambda x: x[1], reverse=True)[:top]

    table = Table(title=f"Elo Ratings — {sector.upper()}", box=box.SIMPLE)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Team", style="bold")
    table.add_column("Elo", justify="right", style="cyan")
    table.add_column("Games", justify="right", style="dim")

    for rank, (team, elo) in enumerate(sorted_teams, 1):
        games = agent._get_count(sector, team)
        table.add_row(str(rank), team.title(), f"{elo:.0f}", str(games))

    console.print(table)


@app.command("update")
def update_result(
    sector: str = typer.Option(..., "--sector", "-s", help="Sector"),
    team_a: str = typer.Option(..., "--home", help="Home / outcome_a team name"),
    team_b: str = typer.Option(..., "--away", help="Away / outcome_b team name"),
    score_a: float = typer.Option(..., "--score-home", help="Final score for home team"),
    score_b: float = typer.Option(..., "--score-away", help="Final score for away team"),
    date: Optional[str] = typer.Option(None, "--date", help="Game date ISO (YYYY-MM-DD)"),
) -> None:
    """Feed a completed game result into all model agents (updates Elo + Form + Poisson)."""
    from evmax.agents.coordinator import AgentCoordinator
    c = AgentCoordinator(sectors=[sector], enable_models=True)
    c.update_models(team_a, team_b, score_a, score_b, sector, date)
    console.print(
        f"[green]Updated models:[/green] {team_a} {score_a:.0f} – {score_b:.0f} {team_b} ({sector})"
    )
