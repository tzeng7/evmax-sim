"""evmax update commands — fetch ESPN scores and update model ratings.

Usage:
  evmax update scores [--date YYYY-MM-DD] [--sectors ...] [--dry-run]
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("scores")
def update_scores(
    target_date: Optional[str] = typer.Option(
        None,
        "--date",
        "-d",
        help="Date to fetch scores for (YYYY-MM-DD). Defaults to yesterday.",
    ),
    sectors: str = typer.Option(
        "soccer,nba,wnba,nfl,ncaab,nhl,baseball",
        "--sectors",
        "-s",
        help=(
            "Comma-separated sectors to update. ESPN-supported: "
            "soccer, nba, wnba, nfl, ncaab, ncaaw, nhl, baseball."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be updated without actually updating models.",
    ),
) -> None:
    """Fetch ESPN completed scores and feed results into Elo/Form/Poisson models."""
    if target_date:
        try:
            fetch_date = date.fromisoformat(target_date)
        except ValueError:
            console.print(f"[red]Invalid date:[/red] {target_date!r} — use YYYY-MM-DD")
            raise typer.Exit(1)
    else:
        fetch_date = date.today() - timedelta(days=1)

    sector_list = [s.strip().lower() for s in sectors.split(",") if s.strip()]

    console.print(
        f"\n[bold cyan]evmax update scores[/bold cyan] — "
        f"date: {fetch_date}  sectors: {', '.join(sector_list)}"
        + ("  [yellow](dry run)[/yellow]" if dry_run else "")
        + "\n"
    )

    asyncio.run(_run_update(sector_list, fetch_date, dry_run))


async def _run_update(
    sector_list: list[str],
    fetch_date: date,
    dry_run: bool,
) -> None:
    from evmax.agents.cleanup.resolver import fetch_completed_scores, _slug_teams, _fuzzy_team_match
    from evmax.agents.cleanup.db import get_connection
    from evmax.agents.coordinator import AgentCoordinator

    FUZZY_THRESHOLD = 65

    table = Table(
        title=f"Model Update Results — {fetch_date}",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Sector", style="dim", width=7)
    table.add_column("Home", min_width=20)
    table.add_column("Away", min_width=20)
    table.add_column("Score", justify="center", width=9)
    table.add_column("Status", justify="center", width=12)

    coordinator = AgentCoordinator(sectors=sector_list, enable_models=True)
    total_updated = 0

    for sector in sector_list:
        scores = await fetch_completed_scores(sector, fetch_date)
        if not scores:
            console.print(f"[dim]  {sector.upper()}: no ESPN scores found[/dim]")
            continue

        # Load event_ids from DB for this sector+date to get canonical team names
        conn = get_connection()
        db_rows = conn.execute(
            """SELECT DISTINCT event_id FROM ev_predictions
               WHERE sector = ? AND event_date = ?""",
            (sector, fetch_date.isoformat()),
        ).fetchall()
        conn.close()

        # Also process all ESPN scores even without DB match (updates general Elo)
        # Match ESPN game → our stored event_id slug for team name extraction
        processed: set[tuple[str, str]] = set()

        # Sector-aware normalizer for the no-DB-match fallback. Without this
        # the fallback was `home_n.lower()` ("indiana fever"), which split
        # state into two buckets for the same team — one keyed by the
        # canonical slug ("fever", from prior seeds) and one keyed by the
        # raw ESPN name. Caught after a WNBA backfill produced duplicate
        # entries for nearly every team.
        from evmax.matching.normalizer import NameNormalizer
        _normalizer = NameNormalizer(sector)

        for score in scores:
            home_n = score["home_name"]
            away_n = score["away_name"]
            home_s = int(score["home_score"])
            away_s = int(score["away_score"])

            # Find best-matching DB event to get slug-based team names
            team_a = _normalizer.normalize(home_n) or home_n.lower()
            team_b = _normalizer.normalize(away_n) or away_n.lower()

            for row in db_rows:
                slug_a, slug_b = _slug_teams(row["event_id"])
                if not slug_a:
                    continue
                a_match = (
                    _fuzzy_team_match(slug_a, home_n) >= FUZZY_THRESHOLD
                    and _fuzzy_team_match(slug_b, away_n) >= FUZZY_THRESHOLD
                )
                if a_match:
                    # Use slug names (already normalized, consistent with model state)
                    team_a = slug_a
                    team_b = slug_b
                    break

            key = (team_a, team_b)
            if key in processed:
                continue
            processed.add(key)

            score_str = f"{home_s}–{away_s}"
            status_str = "[green]updated[/green]" if not dry_run else "[yellow]dry run[/yellow]"

            table.add_row(
                sector.upper(),
                home_n[:22],
                away_n[:22],
                score_str,
                status_str,
            )

            if not dry_run:
                try:
                    coordinator.update_models(
                        team_a=team_a,
                        team_b=team_b,
                        score_a=home_s,
                        score_b=away_s,
                        sector=sector,
                        event_date=fetch_date.isoformat(),
                    )
                    total_updated += 1

                    # Feed shot stats into xG agent (soccer only)
                    if sector == "soccer":
                        home_sot = score.get("home_sot")
                        away_sot = score.get("away_sot")
                        home_shots = score.get("home_shots")
                        away_shots = score.get("away_shots")
                        if all(v is not None for v in (home_sot, away_sot, home_shots, away_shots)):
                            xg = coordinator.soccer_xg_agent
                            xg.record_match(
                                team=team_a, goals_for=home_s, goals_against=away_s,
                                shots_on_target=home_sot, total_shots=home_shots,
                                opponent_sot=away_sot, opponent_shots=away_shots,
                                match_date=fetch_date.isoformat(), is_home=True,
                            )
                            xg.record_match(
                                team=team_b, goals_for=away_s, goals_against=home_s,
                                shots_on_target=away_sot, total_shots=away_shots,
                                opponent_sot=home_sot, opponent_shots=home_shots,
                                match_date=fetch_date.isoformat(), is_home=False,
                            )
                            xg.save_state()
                except Exception as e:
                    console.print(f"[yellow]  Warning: update failed for {team_a} vs {team_b}: {e}[/yellow]")

    console.print(table)
    if not dry_run:
        console.print(f"\n[green]Updated {total_updated} game(s) across {len(sector_list)} sector(s).[/green]")
    else:
        console.print(f"\n[yellow]Dry run — no models updated.[/yellow]")
