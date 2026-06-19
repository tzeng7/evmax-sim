"""CLI commands for standalone point projections.

Commands:
  evmax project game    — project scores for a specific matchup
  evmax project slate   — project all games for a sector from today's Pinnacle slate
  evmax project teams   — list available teams for a sector
  evmax project resolve — resolve logged projections against ESPN actual scores
  evmax project track   — show model accuracy metrics (MAE, ATS record, O/U record)
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evmax.models_ml.point_projection import PointProjectionModel

app = typer.Typer(no_args_is_help=True)
console = Console()

SUPPORTED_SECTORS = ["nba", "nfl", "ncaab", "ncaaw", "soccer"]

# Only NBA uses injury-driven ORTG adjustments for projections today.
INJURY_ENABLED_SECTORS = {"nba"}


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

        # Group by base event (strip ::spread, ::total suffixes)
        events: dict[str, dict] = {}
        for odds in odds_list:
            eid = odds.event_id
            # Extract base event key
            base_key = eid
            for suffix in ("::spread", "::total"):
                if suffix in base_key:
                    base_key = base_key[:base_key.index(suffix)]
                    break

            if base_key not in events:
                events[base_key] = {
                    "home": odds.outcome_a_label,
                    "away": odds.outcome_b_label,
                    "book_spread": None,
                    "book_total": None,
                    "event_date": None,
                }

            if odds.event_date is not None:
                # Convert UTC → US/Eastern so the "game day" matches ESPN and
                # the NBA schedule (late-PT tipoffs don't roll into the next
                # UTC day).
                from zoneinfo import ZoneInfo
                et = odds.event_date.astimezone(ZoneInfo("US/Eastern"))
                events[base_key]["event_date"] = et.date().isoformat()

            if "::spread" in eid and odds.spread_line is not None:
                events[base_key]["book_spread"] = odds.spread_line
            elif "::total" in eid and odds.total_line is not None:
                events[base_key]["book_total"] = odds.total_line

        for key, ev in events.items():
            result = model.project_from_sharp(
                home_team=ev["home"],
                away_team=ev["away"],
                sector=sector,
                book_spread=ev.get("book_spread"),
                book_total=ev.get("book_total"),
                game_date=ev.get("event_date"),
                injury_reports=injury_reports,
            )
            if result:
                # Attach per-event date so logging + display use the real game day
                result["game_date"] = ev.get("event_date") or date.today().isoformat()
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
            # Did the actual result cover the book spread?
            actual_margin = actual_home - actual_away  # positive = home won
            book_spread = row["book_spread"]  # negative = home favored
            # Home covers if margin > |spread|; away covers if margin < spread
            if row["spread_play"].lower() == row["home_team"].lower():
                spread_hit = 1 if actual_margin > abs(book_spread) else 0
            else:
                spread_hit = 1 if actual_margin < -abs(book_spread) else 0

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
