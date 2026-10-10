"""CLI commands for standalone point projections.

Commands:
  evmax project game    — project scores for a specific matchup
  evmax project slate   — project all games for a sector from today's Pinnacle slate
  evmax project teams   — list available teams for a sector
  evmax project resolve — resolve logged projections against ESPN actual scores
  evmax project track   — show model accuracy metrics (MAE, ATS record, O/U record)
  evmax project nfl     — project an NFL week with the nfl_projections game model
  evmax project nfl-run     — project + store an NFL week (games and players), optionally post to Discord
  evmax project nfl-resolve — grade stored NFL projections against final results
  evmax project nfl-track   — tracked accuracy of stored NFL projections
  evmax project nfl-sim     — joint simulation of one NFL game (consistent box scores, stack probabilities)
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evmax.models_ml.point_projection import INJURY_SECTORS, PointProjectionModel

app = typer.Typer(no_args_is_help=True)
console = Console()

SUPPORTED_SECTORS = ["nba", "nfl", "ncaab", "ncaaw", "soccer"]

# Only NBA uses injury-driven ORTG adjustments for projections today.
INJURY_ENABLED_SECTORS = INJURY_SECTORS


async def _fetch_injury_reports(sector: str) -> dict:
    """Fetch injury reports via InjuryReportAgent. Returns {} on any error."""
    if sector not in INJURY_ENABLED_SECTORS:
        return {}
    try:
        from evmax.agents.base import AgentRequest
        from evmax.agents.intelligence.injury_agent import InjuryReportAgent

        agent = InjuryReportAgent()
        resp = await agent.run(AgentRequest(sector=sector, correlation_id="project-cli"))
        return resp.data or {}
    except Exception as e:
        console.print(f"[yellow]Injury fetch failed ({e}); projecting without injury adjustments.[/yellow]")
        return {}

# ---------------------------------------------------------------------------
# Projections database (separate from predictions.db — standalone tool)
# ---------------------------------------------------------------------------

_PROJ_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "projections.db"

_PROJ_SCHEMA = """
CREATE TABLE IF NOT EXISTS projections (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at         TEXT NOT NULL DEFAULT (datetime('now')),
    game_date         TEXT NOT NULL,
    sector            TEXT NOT NULL,
    home_team         TEXT NOT NULL,
    away_team         TEXT NOT NULL,
    proj_home_pts     REAL NOT NULL,
    proj_away_pts     REAL NOT NULL,
    proj_spread       REAL NOT NULL,
    proj_total        REAL NOT NULL,
    book_spread       REAL,
    book_total        REAL,
    spread_play       TEXT,
    total_play         TEXT,
    home_elo          REAL,
    away_elo          REAL,
    confidence        TEXT,
    actual_home_pts   REAL,
    actual_away_pts   REAL,
    spread_hit        INTEGER,
    total_hit         INTEGER,
    resolved_at       TEXT,
    UNIQUE(game_date, sector, home_team, away_team)
);
"""

_INJURY_COLUMN_MIGRATIONS: list[tuple[str, str]] = [
    ("home_ortg_adj", "REAL"),
    ("away_ortg_adj", "REAL"),
    ("home_injury_notes", "TEXT"),
    ("away_injury_notes", "TEXT"),
]


def _get_proj_db() -> sqlite3.Connection:
    _PROJ_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_PROJ_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(_PROJ_SCHEMA)

    # Additive migration for injury columns — safe to run on every open.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(projections)")}
    for col, typ in _INJURY_COLUMN_MIGRATIONS:
        if col not in existing:
            conn.execute(f"ALTER TABLE projections ADD COLUMN {col} {typ}")
    conn.commit()
    return conn


def _log_projection(conn: sqlite3.Connection, result: dict, game_date: str, sector: str) -> bool:
    """Insert or ignore a projection row. Returns True if inserted."""
    proj = result["projection"]
    try:
        conn.execute(
            """INSERT OR IGNORE INTO projections
               (game_date, sector, home_team, away_team,
                proj_home_pts, proj_away_pts, proj_spread, proj_total,
                book_spread, book_total, spread_play, total_play,
                home_elo, away_elo, confidence,
                home_ortg_adj, away_ortg_adj,
                home_injury_notes, away_injury_notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                game_date, sector, proj.home_team, proj.away_team,
                proj.home_points, proj.away_points, proj.projected_spread, proj.projected_total,
                result.get("book_spread"), result.get("book_total"),
                result.get("spread_play"), result.get("total_play"),
                proj.home_elo, proj.away_elo, proj.confidence,
                getattr(proj, "home_ortg_adj", 0.0),
                getattr(proj, "away_ortg_adj", 0.0),
                getattr(proj, "home_injury_notes", None),
                getattr(proj, "away_injury_notes", None),
            ),
        )
        conn.commit()
        return conn.total_changes > 0
    except Exception:
        return False


def _confidence_color(conf: str) -> str:
    return {"high": "green", "medium": "yellow", "low": "red"}.get(conf, "white")


def _spread_str(spread: float) -> str:
    """Format spread as -8.5 or +3.0."""
    if spread == 0:
        return "PK"
    sign = "" if spread < 0 else "+"
    return f"{sign}{spread:.1f}"


def _render_game(result: dict) -> None:
    """Render a single game projection in the style of the Twitter model card."""
    proj = result["projection"]
    book_spread = result.get("book_spread")
    book_total = result.get("book_total")

    # Header
    sector_upper = proj.sector.upper()
    conf_color = _confidence_color(proj.confidence)

    console.print()
    console.print(
        Panel(
            f"[bold]{proj.away_team}[/bold]  @  [bold]{proj.home_team}[/bold]",
            title=f"[bold cyan]{sector_upper}[/bold cyan]",
            subtitle=f"Confidence: [{conf_color}]{proj.confidence.upper()}[/{conf_color}]",
            border_style="cyan",
            width=64,
        )
    )

    # Team projections
    teams_table = Table(box=box.SIMPLE_HEAVY, show_header=True, width=64)
    teams_table.add_column("", style="dim", width=12)
    teams_table.add_column("Away", justify="center", width=24)
    teams_table.add_column("Home", justify="center", width=24)

    teams_table.add_row(
        "Team",
        f"[bold]{proj.away_team}[/bold]",
        f"[bold]{proj.home_team}[/bold]",
    )
    teams_table.add_row(
        "Projected",
        f"[bold yellow]{proj.away_points:.2f}[/bold yellow]",
        f"[bold yellow]{proj.home_points:.2f}[/bold yellow]",
    )
    teams_table.add_row(
        "Elo",
        f"{proj.away_elo:.0f}" if proj.away_elo is not None else "—",
        f"{proj.home_elo:.0f}" if proj.home_elo is not None else "—",
    )
    teams_table.add_row(
        "Win Prob",
        f"{(1 - proj.win_prob_home) * 100:.1f}%",
        f"{proj.win_prob_home * 100:.1f}%",
    )
    console.print(teams_table)

    # Projections vs Book lines
    lines_table = Table(box=box.SIMPLE_HEAVY, show_header=True, width=64)
    lines_table.add_column("", style="dim", width=16)
    lines_table.add_column("Model", justify="center", width=20)
    lines_table.add_column("Sportsbook", justify="center", width=20)

    lines_table.add_row(
        "Spread",
        f"[bold]{_spread_str(proj.projected_spread)}[/bold]  [dim]σ={proj.margin_sigma:.1f}[/dim]",
        _spread_str(book_spread) if book_spread is not None else "—",
    )
    lines_table.add_row(
        "Total",
        f"[bold]{proj.projected_total:.1f}[/bold]  [dim]σ={proj.total_sigma:.1f}[/dim]",
        f"{book_total:.1f}" if book_total is not None else "—",
    )
    console.print(lines_table)
    console.print(f"[dim]Engine: {proj.engine}[/dim]")

    # Injury note (only renders when someone is out/DTD on either team)
    home_adj = getattr(proj, "home_ortg_adj", 0.0) or 0.0
    away_adj = getattr(proj, "away_ortg_adj", 0.0) or 0.0
    if home_adj or away_adj:
        inj_table = Table(box=box.SIMPLE_HEAVY, show_header=True, width=64, title="[bold]Injury Adjustments[/bold]")
        inj_table.add_column("Team", width=14)
        inj_table.add_column("ORTG Δ", justify="right", width=10)
        inj_table.add_column("Notes", no_wrap=False, min_width=36)
        if home_adj:
            inj_table.add_row(
                proj.home_team,
                f"[red]{home_adj:+.1f}[/red]",
                (proj.home_injury_notes or "") or "—",
            )
        if away_adj:
            inj_table.add_row(
                proj.away_team,
                f"[red]{away_adj:+.1f}[/red]",
                (proj.away_injury_notes or "") or "—",
            )
        console.print(inj_table)

    # Recommended plays
    spread_play = result.get("spread_play")
    total_play = result.get("total_play")
    spread_edge = result.get("spread_edge")
    total_edge = result.get("total_edge")

    if spread_play or total_play:
        plays_table = Table(
            box=box.SIMPLE_HEAVY, show_header=True, width=64,
            title="[bold]Recommended Plays[/bold]",
        )
        plays_table.add_column("Market", width=20)
        plays_table.add_column("Play", width=20)
        plays_table.add_column("Edge", justify="right", width=16)

        if spread_play:
            plays_table.add_row(
                "Spread",
                f"[bold green]{spread_play}[/bold green]",
                f"{abs(spread_edge):.1f} pts" if spread_edge else "",
            )
        if total_play:
            plays_table.add_row(
                "Total",
                f"[bold green]{total_play}[/bold green]",
                f"{abs(total_edge):.1f} pts" if total_edge else "",
            )
        console.print(plays_table)
    else:
        if book_spread is not None or book_total is not None:
            console.print("[dim]  No actionable edge detected.[/dim]")


@app.command()
def game(
    home: str = typer.Argument(..., help="Home team name (e.g. 'celtics')"),
    away: str = typer.Argument(..., help="Away team name (e.g. 'hawks')"),
    sector: Optional[str] = typer.Option(None, "--sector", "-s", help="Sport sector (auto-detected if omitted)"),
    book_spread: Optional[float] = typer.Option(None, "--spread", help="Sportsbook home spread (e.g. -7.5)"),
    book_total: Optional[float] = typer.Option(None, "--total", help="Sportsbook total (e.g. 146.5)"),
    game_date: Optional[str] = typer.Option(None, "--date", "-d", help="Game date (YYYY-MM-DD). Enables NBA playoff tightening."),
    injuries: bool = typer.Option(True, "--injuries/--no-injuries", help="Apply ESPN injury ORTG adjustments (NBA only)."),
) -> None:
    """Project scores for a specific matchup."""
    model = PointProjectionModel()

    if sector is None:
        # Auto-detect from team names
        detected = model.detect_sector(home) or model.detect_sector(away)
        if detected:
            sector = detected
            console.print(f"[dim]Auto-detected sector: {sector.upper()}[/dim]")
        else:
            console.print("[red]Could not detect sector from team names. Use --sector.[/red]")
            raise typer.Exit(1)
    else:
        sector = sector.lower()

    if sector not in SUPPORTED_SECTORS:
        console.print(f"[red]Unsupported sector: {sector}. Use one of: {', '.join(SUPPORTED_SECTORS)}[/red]")
        raise typer.Exit(1)

    injury_reports = asyncio.run(_fetch_injury_reports(sector)) if injuries else {}

    result = model.project_from_sharp(
        home_team=home,
        away_team=away,
        sector=sector,
        book_spread=book_spread,
        book_total=book_total,
        game_date=game_date,
        injury_reports=injury_reports,
    )

    if result is None:
        console.print("[red]Could not generate projection. Check team names.[/red]")
        console.print(f"[dim]Available teams: evmax project teams --sector {sector}[/dim]")
        raise typer.Exit(1)

    _render_game(result)


@app.command()
def slate(
    sector: str = typer.Option("nba", "--sector", "-s", help="Sport sector"),
    log: bool = typer.Option(False, "--log", help="Save projections to projections.db for tracking."),
    injuries: bool = typer.Option(True, "--injuries/--no-injuries", help="Apply ESPN injury ORTG adjustments (NBA only)."),
) -> None:
    """Project all games on today's Pinnacle slate for a sector."""
    sector = sector.lower()
    if sector not in SUPPORTED_SECTORS:
        console.print(f"[red]Unsupported sector: {sector}. Use one of: {', '.join(SUPPORTED_SECTORS)}[/red]")
        raise typer.Exit(1)

    async def _run() -> list[dict]:
        from evmax.clients.esports_pinnacle import PinnacleGuestClient
        from evmax.projections.point import build_slate

        injury_reports: dict = {}
        if injuries:
            injury_reports = await _fetch_injury_reports(sector)
            if injury_reports:
                teams_flagged = sum(1 for r in injury_reports.values() if getattr(r, "players", None))
                console.print(f"[dim]Loaded injuries for {teams_flagged} teams.[/dim]")

        async with PinnacleGuestClient() as client:
            odds_list = await client.get_odds(sector)

        model = PointProjectionModel()
        results = []
        # One entry per game: home/away from the moneyline, the spread as the HOME
        # handicap, Pinnacle's main total (build_slate documents the conventions).
        for ev in build_slate(odds_list):
            result = model.project_from_sharp(
                home_team=ev.home,
                away_team=ev.away,
                sector=sector,
                book_spread=ev.book_spread,
                book_total=ev.book_total,
                game_date=ev.game_date,
                injury_reports=injury_reports,
            )
            if result:
                # Attach per-event date so logging + display use the real game day
                result["game_date"] = ev.game_date or date.today().isoformat()
                results.append(result)

        return results

    results = asyncio.run(_run())

    if not results:
        console.print(f"[yellow]No games found for {sector.upper()} on today's slate.[/yellow]")
        raise typer.Exit()

    # Log projections if requested
    if log:
        conn = _get_proj_db()
        fallback = date.today().isoformat()
        logged = sum(
            1 for r in results
            if _log_projection(conn, r, r.get("game_date") or fallback, sector)
        )
        conn.close()
        console.print(f"[green]Logged {logged} new projections to projections.db[/green]")

    console.print(f"\n[bold cyan]{sector.upper()} Slate — {len(results)} games[/bold cyan]\n")

    # Summary table
    summary = Table(box=box.ROUNDED, show_header=True, title="Point Projections")
    summary.add_column("Matchup", no_wrap=False, min_width=28)
    summary.add_column("Away Pts", justify="right", width=9)
    summary.add_column("Home Pts", justify="right", width=9)
    summary.add_column("Proj Spread", justify="right", width=12)
    summary.add_column("Book Spread", justify="right", width=12)
    summary.add_column("Proj Total", justify="right", width=11)
    summary.add_column("Book Total", justify="right", width=11)
    summary.add_column("Inj", justify="right", width=10)
    summary.add_column("Plays", no_wrap=False, width=18)
    summary.add_column("Conf", width=6)

    for r in sorted(results, key=lambda x: abs(x.get("spread_edge") or 0), reverse=True):
        proj = r["projection"]
        plays = []
        if r.get("spread_play"):
            plays.append(f"{r['spread_play']} ({abs(r['spread_edge']):.1f})")
        if r.get("total_play"):
            plays.append(f"{r['total_play']} ({abs(r['total_edge']):.1f})")

        conf_color = _confidence_color(proj.confidence)

        home_adj = getattr(proj, "home_ortg_adj", 0.0) or 0.0
        away_adj = getattr(proj, "away_ortg_adj", 0.0) or 0.0
        if home_adj or away_adj:
            inj_str = f"H{home_adj:+.0f}/A{away_adj:+.0f}"
        else:
            inj_str = "[dim]—[/dim]"

        summary.add_row(
            f"{proj.away_team} @ {proj.home_team}",
            f"{proj.away_points:.1f}",
            f"{proj.home_points:.1f}",
            _spread_str(proj.projected_spread),
            _spread_str(r["book_spread"]) if r.get("book_spread") is not None else "—",
            f"{proj.projected_total:.1f}",
            f"{r['book_total']:.1f}" if r.get("book_total") is not None else "—",
            inj_str,
            ", ".join(plays) if plays else "[dim]—[/dim]",
            f"[{conf_color}]{proj.confidence[0].upper()}[/{conf_color}]",
        )

    console.print(summary)
    console.print()

    # Detailed view for games with plays
    actionable = [r for r in results if r.get("spread_play") or r.get("total_play")]
    if actionable:
        console.print(f"[bold]Detailed projections for {len(actionable)} actionable games:[/bold]")
        for r in actionable:
            _render_game(r)


@app.command()
def teams(
    sector: str = typer.Option("ncaab", "--sector", "-s", help="Sport sector"),
) -> None:
    """List available teams with model data for a sector."""
    model = PointProjectionModel()
    team_list = model.available_teams(sector.lower())

    if not team_list:
        console.print(f"[yellow]No teams found for {sector.upper()}.[/yellow]")
        raise typer.Exit()

    console.print(f"\n[bold]{sector.upper()}[/bold] — {len(team_list)} teams with Poisson data:\n")

    # Display in columns
    cols = 4
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    for _ in range(cols):
        table.add_column(width=24)

    for i in range(0, len(team_list), cols):
        chunk = team_list[i : i + cols]
        while len(chunk) < cols:
            chunk.append("")
        table.add_row(*chunk)

    console.print(table)


# ---------------------------------------------------------------------------
# ESPN score fetching (reused from resolver pattern)
# ---------------------------------------------------------------------------

ESPN_SPORT_MAP: dict[str, tuple[str, str, dict]] = {
    "nba": ("basketball", "nba", {}),
    "ncaab": ("basketball", "mens-college-basketball", {"groups": "50"}),
    "ncaaw": ("basketball", "womens-college-basketball", {"groups": "50"}),
    "nfl": ("football", "nfl", {}),
}

ESPN_SOCCER_LEAGUES = ["eng.1", "esp.1", "ger.1", "ita.1", "fra.1", "uefa.champions"]


async def _fetch_espn_scores(sector: str, target_date: str) -> list[dict]:
    """Fetch completed game scores from ESPN for a date (YYYY-MM-DD)."""
    import httpx

    espn_date = target_date.replace("-", "")
    results = []

    async with httpx.AsyncClient(timeout=15) as client:
        if sector == "soccer":
            for league in ESPN_SOCCER_LEAGUES:
                url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
                try:
                    r = await client.get(url, params={"dates": espn_date, "limit": 200})
                    r.raise_for_status()
                    results.extend(_parse_espn_events(r.json()))
                except Exception:
                    continue
        else:
            sport, league, extra = ESPN_SPORT_MAP.get(sector, ("basketball", "nba", {}))
            url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
            params = {"dates": espn_date, "limit": 200}
            params.update(extra)
            try:
                r = await client.get(url, params=params)
                r.raise_for_status()
                results = _parse_espn_events(r.json())
            except Exception:
                pass

    return results


def _parse_espn_events(data: dict) -> list[dict]:
    results = []
    for event in data.get("events", []):
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        if not comp.get("status", {}).get("type", {}).get("completed"):
            continue
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        try:
            results.append({
                "home_name": home.get("team", {}).get("displayName", ""),
                "away_name": away.get("team", {}).get("displayName", ""),
                "home_score": int(home.get("score", 0)),
                "away_score": int(away.get("score", 0)),
            })
        except (ValueError, TypeError):
            continue
    return results


def _spread_hit(play_is_home: bool, home_margin: float, home_handicap: float) -> int:
    """1 if the play covered ``home_handicap`` (the HOME line, negative = home favored), else 0.

    Home covers when home_margin + home_handicap > 0; away covers when it is < 0.
    A push grades 0, as before. (Until 2026-10-10 this assumed the home team was
    always the favorite: away plays had to win outright by the spread and home
    underdogs had to win by it.)
    """
    net = home_margin + home_handicap
    return 1 if (net > 0 if play_is_home else net < 0) else 0


def _fuzzy_match_team(name: str, candidates: list[str], threshold: int = 72) -> Optional[str]:
    """Find best fuzzy match for a team name."""
    from rapidfuzz import fuzz

    name_lower = name.lower().strip()
    best_score = 0
    best_match = None
    for c in candidates:
        score = fuzz.token_sort_ratio(name_lower, c.lower())
        if score > best_score and score >= threshold:
            best_score = score
            best_match = c
    return best_match


@app.command()
def resolve(
    game_date: str = typer.Option(
        (date.today() - timedelta(days=1)).isoformat(),
        "--date", "-d",
        help="Date to resolve (YYYY-MM-DD, default: yesterday).",
    ),
    sector: Optional[str] = typer.Option(None, "--sector", "-s", help="Filter by sector."),
) -> None:
    """Resolve logged projections against actual ESPN scores."""
    conn = _get_proj_db()

    # Find unresolved projections for the date
    query = "SELECT * FROM projections WHERE game_date = ? AND resolved_at IS NULL"
    params: list = [game_date]
    if sector:
        query += " AND sector = ?"
        params.append(sector.lower())

    rows = conn.execute(query, params).fetchall()
    if not rows:
        console.print(f"[yellow]No unresolved projections for {game_date}.[/yellow]")
        conn.close()
        raise typer.Exit()

    # Group by sector
    sectors_needed = set(r["sector"] for r in rows)

    async def _fetch_all() -> dict[str, list[dict]]:
        results = {}
        for s in sectors_needed:
            results[s] = await _fetch_espn_scores(s, game_date)
        return results

    scores_by_sector = asyncio.run(_fetch_all())

    resolved = 0
    for row in rows:
        scores = scores_by_sector.get(row["sector"], [])
        if not scores:
            continue

        # Try to match home team
        espn_names = [s["home_name"] for s in scores] + [s["away_name"] for s in scores]
        home_match = _fuzzy_match_team(row["home_team"], [s["home_name"] for s in scores])
        if not home_match:
            continue

        game = next((s for s in scores if s["home_name"] == home_match), None)
        if not game:
            continue

        actual_home = game["home_score"]
        actual_away = game["away_score"]
        actual_spread = -(actual_home - actual_away)  # negative = home won by X
        actual_total = actual_home + actual_away

        # Grade spread play
        spread_hit = None
        if row["book_spread"] is not None and row["spread_play"]:
            spread_hit = _spread_hit(
                play_is_home=row["spread_play"].lower() == row["home_team"].lower(),
                home_margin=actual_home - actual_away,
                home_handicap=row["book_spread"],
            )

        # Grade total play
        total_hit = None
        if row["book_total"] is not None and row["total_play"]:
            if row["total_play"] == "Over":
                total_hit = 1 if actual_total > row["book_total"] else 0
            else:
                total_hit = 1 if actual_total < row["book_total"] else 0

        conn.execute(
            """UPDATE projections SET
               actual_home_pts = ?, actual_away_pts = ?,
               spread_hit = ?, total_hit = ?,
               resolved_at = datetime('now')
               WHERE id = ?""",
            (actual_home, actual_away, spread_hit, total_hit, row["id"]),
        )
        resolved += 1

    conn.commit()
    conn.close()

    console.print(f"[green]Resolved {resolved}/{len(rows)} projections for {game_date}.[/green]")
    if resolved < len(rows):
        console.print(f"[yellow]{len(rows) - resolved} games could not be matched to ESPN results.[/yellow]")


@app.command()
def track(
    days: int = typer.Option(30, "--days", "-d", help="Look back N days."),
    sector: Optional[str] = typer.Option(None, "--sector", "-s", help="Filter by sector."),
) -> None:
    """Show model accuracy metrics for resolved projections."""
    conn = _get_proj_db()

    since = (date.today() - timedelta(days=days)).isoformat()
    query = "SELECT * FROM projections WHERE resolved_at IS NOT NULL AND game_date >= ?"
    params: list = [since]
    if sector:
        query += " AND sector = ?"
        params.append(sector.lower())
    query += " ORDER BY game_date DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        console.print("[yellow]No resolved projections found.[/yellow]")
        console.print("[dim]Run 'evmax project slate --log' to log projections, then 'evmax project resolve' after games complete.[/dim]")
        raise typer.Exit()

    # Compute metrics
    home_errors = []
    away_errors = []
    spread_errors = []
    total_errors = []
    spread_picks = {"wins": 0, "losses": 0, "total": 0}
    total_picks = {"wins": 0, "losses": 0, "total": 0}

    for r in rows:
        if r["actual_home_pts"] is not None:
            home_errors.append(abs(r["proj_home_pts"] - r["actual_home_pts"]))
            away_errors.append(abs(r["proj_away_pts"] - r["actual_away_pts"]))
            actual_spread = -(r["actual_home_pts"] - r["actual_away_pts"])
            actual_total = r["actual_home_pts"] + r["actual_away_pts"]
            spread_errors.append(abs(r["proj_spread"] - actual_spread))
            total_errors.append(abs(r["proj_total"] - actual_total))

        if r["spread_hit"] is not None:
            spread_picks["total"] += 1
            if r["spread_hit"] == 1:
                spread_picks["wins"] += 1
            else:
                spread_picks["losses"] += 1

        if r["total_hit"] is not None:
            total_picks["total"] += 1
            if r["total_hit"] == 1:
                total_picks["wins"] += 1
            else:
                total_picks["losses"] += 1

    n = len(home_errors)
    sector_label = sector.upper() if sector else "ALL SECTORS"

    # Header
    console.print(f"\n[bold cyan]Point Projection Model — {sector_label}[/bold cyan]")
    console.print(f"[dim]{n} resolved games over last {days} days[/dim]\n")

    # Accuracy metrics
    if n > 0:
        metrics = Table(box=box.ROUNDED, title="Accuracy Metrics (MAE)")
        metrics.add_column("Metric", width=24)
        metrics.add_column("Value", justify="right", width=12)

        home_mae = sum(home_errors) / n
        away_mae = sum(away_errors) / n
        spread_mae = sum(spread_errors) / n
        total_mae = sum(total_errors) / n

        metrics.add_row("Home Points MAE", f"{home_mae:.1f} pts")
        metrics.add_row("Away Points MAE", f"{away_mae:.1f} pts")
        metrics.add_row("Spread MAE", f"{spread_mae:.1f} pts")
        metrics.add_row("Total MAE", f"{total_mae:.1f} pts")
        metrics.add_row("Sample Size", str(n))
        console.print(metrics)
        console.print()

    # ATS / O/U record
    record = Table(box=box.ROUNDED, title="Pick Record (vs. Book Lines)")
    record.add_column("Market", width=16)
    record.add_column("W", justify="right", width=6)
    record.add_column("L", justify="right", width=6)
    record.add_column("Win %", justify="right", width=8)
    record.add_column("Edge", justify="right", width=10)

    for label, picks in [("Spread (ATS)", spread_picks), ("Total (O/U)", total_picks)]:
        if picks["total"] > 0:
            wp = picks["wins"] / picks["total"]
            # Edge over 50% (breakeven for -110 is ~52.4%)
            edge = wp - 0.524
            edge_color = "green" if edge > 0 else "red"
            record.add_row(
                label,
                str(picks["wins"]),
                str(picks["losses"]),
                f"{wp * 100:.1f}%",
                f"[{edge_color}]{edge * 100:+.1f}%[/{edge_color}]",
            )
        else:
            record.add_row(label, "—", "—", "—", "—")

    console.print(record)
    console.print()

    # Recent results detail
    recent = rows[:15]
    detail = Table(box=box.SIMPLE, title=f"Recent Projections (last {min(15, len(rows))})")
    detail.add_column("Date", width=10)
    detail.add_column("Matchup", no_wrap=False, min_width=24)
    detail.add_column("Proj", justify="right", width=10)
    detail.add_column("Actual", justify="right", width=10)
    detail.add_column("Spread", justify="center", width=8)
    detail.add_column("Total", justify="center", width=8)

    for r in recent:
        if r["actual_home_pts"] is None:
            continue
        proj_score = f"{r['proj_away_pts']:.0f}-{r['proj_home_pts']:.0f}"
        actual_score = f"{int(r['actual_away_pts'])}-{int(r['actual_home_pts'])}"

        spread_icon = ""
        if r["spread_hit"] == 1:
            spread_icon = "[green]W[/green]"
        elif r["spread_hit"] == 0:
            spread_icon = "[red]L[/red]"
        else:
            spread_icon = "[dim]—[/dim]"

        total_icon = ""
        if r["total_hit"] == 1:
            total_icon = "[green]W[/green]"
        elif r["total_hit"] == 0:
            total_icon = "[red]L[/red]"
        else:
            total_icon = "[dim]—[/dim]"

        detail.add_row(
            r["game_date"],
            f"{r['away_team']} @ {r['home_team']}",
            proj_score,
            actual_score,
            spread_icon,
            total_icon,
        )

    console.print(detail)


# ---------------------------------------------------------------------------
# NFL — evmax.nfl_projections (walk-forward validated, no market inputs)
# ---------------------------------------------------------------------------

def _nfl_line(home: str, away: str, home_margin: float) -> str:
    """Favorite-perspective line from a home margin: 'DAL -3.1', 'TB -2.0' or 'PK'."""
    from evmax.nfl_projections.store import favorite_line

    return favorite_line(home, away, home_margin)


@app.command()
def nfl(
    season: Optional[int] = typer.Option(None, "--season", help="NFL season (default: the next unplayed week's season)."),
    week: Optional[int] = typer.Option(None, "--week", "-w", help="Week (default: the next week with an unplayed game)."),
    refresh: bool = typer.Option(True, "--refresh/--no-refresh", help="Re-download the current season's nflverse data if stale."),
    players: bool = typer.Option(False, "--players", help="Project player stat lines instead of game scores."),
    team: Optional[str] = typer.Option(None, "--team", "-t", help="With --players: only this team (abbreviation, e.g. KC)."),
    log: bool = typer.Option(False, "--log", help="Store the projections in projections.db (updated until kickoff)."),
    espn: bool = typer.Option(True, "--espn/--no-espn", help="With --players: also drop players the live ESPN injury feed lists as out."),
) -> None:
    """Project every game of an NFL week: score, spread, total, win probability.

    With --players: each likely-active skill player's median receptions,
    receiving / rushing / passing yards with a 10th-90th percentile range
    (walk-forward 2019-24: 6-13% lower MAE than a last-8-games average).

    The model (evmax.nfl_projections) reads no market price; the Market column
    shows the nflverse consensus line for comparison only. Walk-forward
    2020-25: margin MAE 10.13 (Vegas close 9.76), total MAE 10.49 (10.28).
    Data cache: EVMAX_NFL_PROJ_DATA (default data/backtest/nfl_projections).
    """
    from evmax.agents.models.nfl_efficiency_agent import NFL_ABBREV_TO_NAME
    from evmax.nfl_projections import data as nfl_data
    from evmax.nfl_projections import live

    if refresh:
        nfl_data.ensure_games()
    games = nfl_data.load_games()
    if season is None or week is None:
        s, w = live.next_week(games, date.today())
        season, week = season or s, week or w
    def full(abbr: str) -> str:
        return NFL_ABBREV_TO_NAME.get(abbr, abbr).title().replace("49Ers", "49ers")

    if players:
        _nfl_players_table(season, week, refresh, team, full, log=log, espn=espn)
        return
    with console.status(f"Projecting NFL {season} week {week}..."):
        df = live.project_week(season, week, refresh=refresh)
    if log:
        from evmax.nfl_projections import store
        from evmax.provenance import code_version

        with store.connect() as conn:
            n = store.log_games(conn, df, code_version())
        console.print(f"[green]Stored {n} game projections (games already kicked off are frozen).[/green]")

    t = Table(box=box.ROUNDED, title=f"NFL {season} Week {week} — projections (model, no market inputs)")
    t.add_column("Kickoff", width=10)
    t.add_column("Event", no_wrap=False, min_width=28)
    t.add_column("Outcome", no_wrap=False, min_width=20)
    t.add_column("Score", justify="right", width=15)
    t.add_column("Home win", justify="right", width=8)
    t.add_column("QBs (away / home)", no_wrap=False, width=24)
    t.add_column("Market", no_wrap=False, width=16)
    for r in df.sort_values(["gameday", "gametime"]).itertuples():
        site = " (neutral)" if r.neutral else ""
        market = "—"
        if pd.notna(r.market_spread_line):
            market = f"{_nfl_line(r.home_team, r.away_team, r.market_spread_line)} · {r.market_total_line:.1f}"
        t.add_row(
            str(r.gameday),
            f"{full(r.away_team)} @ {full(r.home_team)}{site}",
            f"{_nfl_line(r.home_team, r.away_team, r.proj_margin)} · total {r.proj_total:.1f}",
            f"{r.away_team} {r.proj_away:.1f} – {r.home_team} {r.proj_home:.1f}",
            f"{r.p_home_win * 100:.0f}%",
            f"{r.away_qb_name or '?'} / {r.home_qb_name or '?'}",
            market,
        )
    console.print(t)
    console.print("[dim]Outdoor wind uses the league median until a forecast feed is wired; "
                  "starters come from the nflverse schedule (fallback: last game's starter).[/dim]")


def _nfl_players_table(season: int, week: int, refresh: bool, team: Optional[str], full,
                       log: bool = False, espn: bool = True) -> None:
    from evmax.nfl_projections import live

    with console.status(f"Projecting NFL {season} week {week} players..."):
        reports = live.fetch_espn_injury_reports() if espn else None
        df = live.project_week_players(season, week, refresh=refresh, espn_reports=reports)
    if espn and not reports:
        console.print("[yellow]ESPN injury feed unavailable; using the nflverse injury report only.[/yellow]")
    if log:
        from evmax.nfl_projections import store
        from evmax.provenance import code_version

        with store.connect() as conn:
            n = store.log_players(conn, df, code_version())
        console.print(f"[green]Stored {n} player projections (games already kicked off are frozen).[/green]")
    if team:
        df = df[df["team"] == team.upper()]
    df = df[(df["proj_targets"] >= 3) | (df["proj_carries"] >= 5) | df["is_starting_qb"]]
    if df.empty:
        console.print("[yellow]No players to show.[/yellow]")
        return

    def rng(r, stat: str) -> str:
        return f"{getattr(r, 'proj_' + stat):.0f} [dim]({getattr(r, 'p10_' + stat):.0f}–{getattr(r, 'p90_' + stat):.0f})[/dim]"

    t = Table(box=box.ROUNDED, title=f"NFL {season} Week {week} — player projections (median, 10th–90th pct)")
    t.add_column("Event", no_wrap=False, min_width=28)
    t.add_column("Outcome", no_wrap=False, min_width=24)
    t.add_column("Rec", justify="right", width=12)
    t.add_column("Rec yds", justify="right", width=15)
    t.add_column("Rush yds", justify="right", width=15)
    t.add_column("Pass yds", justify="right", width=16)
    t.add_column("TD", justify="right", width=12)
    df = df.sort_values(["gameday", "game_id", "team", "proj_receiving_yards"], ascending=[True, True, True, False])
    for r in df.itertuples():
        event = f"{full(r.away_team)} @ {full(r.home_team)}"
        t.add_row(
            event,
            f"{r.player_display_name} ({r.position}, {r.team})",
            f"{r.proj_receptions:.0f} [dim]({r.p10_receptions:.0f}–{r.p90_receptions:.0f})[/dim]" if r.proj_targets >= 1 else "—",
            rng(r, "receiving_yards") if r.proj_targets >= 1 else "—",
            rng(r, "rushing_yards") if r.proj_carries >= 1 else "—",
            rng(r, "passing_yards") if r.is_starting_qb else "—",
            (f"{r.p_anytime_td * 100:.0f}%" + (f" [dim]· {r.proj_passing_tds:.1f} pass[/dim]" if r.is_starting_qb else ""))
            if "p_anytime_td" in df else "—",
        )
    console.print(t)
    console.print("[dim]Active roster = played in the team's last 3 games minus Out/Doubtful (nflverse injury report "
                  "+ live ESPN feed); teammates absorb 60% of a ruled-out player's targets/carries. "
                  "TD = P(anytime rushing/receiving TD); QBs also show projected passing TDs. "
                  "Medians are MAE-optimal (yardage is right-skewed, so they sit below the mean).[/dim]")


def _resolve_week(season: Optional[int], week: Optional[int]) -> tuple[int, int]:
    from evmax.nfl_projections import data as nfl_data
    from evmax.nfl_projections import live

    if season is not None and week is not None:
        return season, week
    s, w = live.next_week(nfl_data.load_games(), date.today())
    return season or s, week or w


@app.command("nfl-run")
def nfl_run(
    season: Optional[int] = typer.Option(None, "--season", help="NFL season (default: the next unplayed week's season)."),
    week: Optional[int] = typer.Option(None, "--week", "-w", help="Week (default: the next week with an unplayed game)."),
    refresh: bool = typer.Option(True, "--refresh/--no-refresh", help="Re-download stale nflverse data first."),
    espn: bool = typer.Option(True, "--espn/--no-espn", help="Also apply the live ESPN injury feed."),
    resolve: bool = typer.Option(True, "--resolve/--no-resolve", help="Grade finished games before projecting."),
    post: bool = typer.Option(False, "--post", help="Post the stored week to the Discord channel/DM."),
) -> None:
    """Project and store an NFL week (games + players); the scheduled-task entry point.

    Re-running before kickoff refreshes the stored rows (the Sunday-morning run
    picks up late inactives); games that already kicked off stay frozen.
    """
    from evmax.nfl_projections import data as nfl_data
    from evmax.nfl_projections import pipeline, store

    if refresh:
        nfl_data.ensure_games()
    season, week = _resolve_week(season, week)
    with store.connect() as conn:
        if resolve:
            with console.status("Grading finished games..."):
                graded = pipeline.resolve_pending(conn, refresh=refresh)
            console.print(f"Graded {graded['games']} games and {graded['players']} player rows.")
        with console.status(f"Projecting NFL {season} week {week}..."):
            run = pipeline.run_week(conn, season, week, refresh=refresh, espn=espn)
        console.print(f"[green]NFL {season} week {week}: stored {run.games_logged} games and "
                      f"{run.players_logged} player rows.[/green]")
        for note in run.notes:
            console.print(f"[yellow]{note}[/yellow]")
        if post:
            ok = pipeline.post_week(conn, season, week)
            console.print("[green]Posted to Discord.[/green]" if ok
                          else "[yellow]Discord post skipped (bot not configured) or failed.[/yellow]")


@app.command("nfl-resolve")
def nfl_resolve(
    refresh: bool = typer.Option(True, "--refresh/--no-refresh", help="Re-download nflverse results first."),
) -> None:
    """Grade stored NFL projections whose games are final (scores, closing lines, player stats)."""
    from evmax.nfl_projections import pipeline, store

    with store.connect() as conn, console.status("Grading finished games..."):
        n = pipeline.resolve_pending(conn, refresh=refresh)
    console.print(f"Graded {n['games']} games and {n['players']} player rows.")


@app.command("nfl-track")
def nfl_track(
    season: Optional[int] = typer.Option(None, "--season", help="Only this season."),
) -> None:
    """Tracked accuracy of stored, graded NFL projections vs results and the closing line."""
    from evmax.nfl_projections import store

    with store.connect() as conn:
        acc = store.accuracy(conn, season=season)
    g = acc["games"]
    scope = f"season {season}" if season else "all seasons"
    if not g["n"] and not acc["players"]:
        console.print(f"[yellow]No graded NFL projections ({scope}). Run `evmax project nfl-run` before games "
                      f"and `evmax project nfl-resolve` after.[/yellow]")
        return
    t = Table(box=box.ROUNDED, title=f"NFL projections — tracked accuracy ({scope})")
    t.add_column("Event", no_wrap=False, min_width=24)
    t.add_column("Outcome", no_wrap=False, min_width=24)
    t.add_column("n", justify="right")
    t.add_column("Model MAE", justify="right")
    t.add_column("Closing line MAE", justify="right")
    t.add_column("Bias", justify="right")
    t.add_column("< p10 / <= p90", justify="right")
    if g["n"]:
        def close(v):
            return f"{v:.2f}" if v is not None else "—"
        t.add_row("All graded games", "Margin (home - away)", str(g["n"]), f"{g['margin_mae']:.2f}",
                  close(g["close_margin_mae"]), "", "")
        t.add_row("All graded games", "Total points", str(g["n"]), f"{g['total_mae']:.2f}",
                  close(g["close_total_mae"]), "", "")
        if g.get("winner_pct") is not None:
            t.add_row("All graded games", "Winner picked", str(g["n"]), f"{g['winner_pct'] * 100:.1f}% right", "", "", "")
    labels = {"receptions": "Receptions (proj targets >= 3)", "receiving_yards": "Receiving yards (proj targets >= 3)",
              "rushing_yards": "Rushing yards (proj carries >= 5)", "passing_yards": "Passing yards (starting QB)"}
    for st, m in acc["players"].items():
        if st == "anytime_td":
            t.add_row("Players who played", "Anytime TD (receiving + rushing populations)", str(m["n"]),
                      f"Brier {m['brier']:.4f}", "—", f"{(m['mean_p'] - m['rate']) * 100:+.1f}pp", "")
            continue
        if st == "passing_tds":
            t.add_row("Players who played", "Passing TDs (starting QB)", str(m["n"]), f"{m['mae']:.2f}", "—",
                      f"{m['bias']:+.2f}", "")
            continue
        t.add_row("Players who played", labels[st], str(m["n"]), f"{m['mae']:.2f}", "—", f"{m['bias']:+.2f}",
                  f"{m['below_p10'] * 100:.0f}% / {m['at_or_below_p90'] * 100:.0f}%")
    console.print(t)
    console.print("[dim]Ideal range coverage: 10% below p10, 90% at or below p90. "
                  "The closing line is the nflverse consensus after the game.[/dim]")


@app.command("nfl-sim")
def nfl_sim(
    team: str = typer.Option(..., "--team", "-t", help="Either team of the game, e.g. KC."),
    season: Optional[int] = typer.Option(None, "--season"),
    week: Optional[int] = typer.Option(None, "--week", "-w"),
    sims: int = typer.Option(10000, "--sims", help="Simulated games."),
    refresh: bool = typer.Option(True, "--refresh/--no-refresh"),
    espn: bool = typer.Option(True, "--espn/--no-espn"),
) -> None:
    """Simulate one NFL game's box score jointly (evmax.nfl_projections.simulate).

    Receivers' yards add up to the QB's passing yards in every simulation, so
    teammates' lines are correlated the way real games are. The Stacks table
    shows P(QB and receiver both clear their medians) next to the product of
    the separate probabilities (what independent projections would imply).
    Validation (2025 holdout): scripts/eval_nfl_joint_sim.py.
    """
    from evmax.nfl_projections import live, simulate

    season, week = _resolve_week(season, week)
    with console.status(f"Simulating {team.upper()} in NFL {season} week {week} ({sims:,} games)..."):
        reports = live.fetch_espn_injury_reports() if espn else None
        res = live.simulate_game(season, week, team, n=sims, refresh=refresh, espn_reports=reports)
    teams = list(res)
    event = " @ ".join(sorted(teams, key=lambda t: int(res[t][0]["home"].iloc[0])))
    t = Table(box=box.ROUNDED, title=f"NFL {season} Week {week} — simulated box score ({sims:,} games)")
    t.add_column("Event", no_wrap=False, min_width=24)
    t.add_column("Outcome", no_wrap=False, min_width=24)
    for col in ("Rec", "Rec yds", "Rush yds", "Pass yds"):
        t.add_column(col, justify="right")
    t.add_column("TD", justify="right")
    stacks = []
    for tm in teams:
        players, s = res[tm]
        summ = simulate.summarize(players, s)
        show = (players["proj_targets"] >= 2.5) | (players["proj_carries"] >= 5) | players["is_starting_qb"]
        order = players[show].sort_values(["is_starting_qb", "proj_targets"], ascending=[False, False]).index
        for i in order:
            p, m = players.loc[i], summ.loc[i]

            def cell(stat: str, ok: bool) -> str:
                if not ok:
                    return "—"
                med, lo, hi = (int(round(m[f"{k}{stat}"])) + 0 for k in ("sim_", "sim_p10_", "sim_p90_"))
                return f"{med} [dim]({lo}–{hi})[/dim]"
            t.add_row(event, f"{p['player_display_name']} ({p['position']}, {tm})",
                      cell("receptions", p["proj_targets"] >= 1), cell("receiving_yards", p["proj_targets"] >= 1),
                      cell("rushing_yards", p["proj_carries"] >= 1), cell("passing_yards", bool(p["is_starting_qb"])),
                      f"{m['sim_p_anytime_td'] * 100:.0f}%")
        for k in simulate.qb_stacks(players, s):
            stacks.append((event, f"{k['qb']} {k['qb_passing_yards']:.0f}+ pass & "
                                  f"{k['receiver']} {k['receiver_receiving_yards']:.0f}+ rec yds",
                           k["joint"], k["independent"]))
    console.print(t)
    if stacks:
        st = Table(box=box.ROUNDED, title="Stacks — both legs clear their simulated medians")
        st.add_column("Event", no_wrap=False, min_width=24)
        st.add_column("Outcome", no_wrap=False, min_width=24)
        st.add_column("Joint", justify="right")
        st.add_column("If independent", justify="right")
        for e, o, j, i in stacks:
            st.add_row(e, o, f"{j * 100:.1f}%", f"{i * 100:.1f}%")
        console.print(st)
    console.print("[dim]Medians with 10th–90th percentile ranges from the simulation. Model only, no market "
                  "inputs; correlations validated on the 2025 holdout (QB–WR1 simulated +0.55 vs realized +0.46).[/dim]")
