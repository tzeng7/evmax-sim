"""End-to-end golf run: data → skill → simulate → price → observe edge.

Wires the whole standalone module together for one or more tournaments. Runs
from captured FIXTURES by default (fully offline — the tested path) or from
LIVE venue/data feeds with ``--live``. This is an observation tool: it surfaces
model-vs-market divergence per market; it does not place bets or touch any
bankroll (golf is not in the scan pipeline).

    python -m evmax.golf.run              # offline, from fixtures
    python -m evmax.golf.run --live       # live PolyUS + PGA SG (+ Kalshi if creds)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

from evmax.golf.data.pga import parse_sg_snapshot
from evmax.golf.devig import devig_field
from evmax.golf.models import FieldPrices, GolfEdge, GolfMarketType, SimResult
from evmax.golf.pricing import EdgeConfig, flag_plays, price_field
from evmax.golf.simulator import SimConfig, simulate_field
from evmax.golf.skill import SkillConfig, skill_from_pga, to_sim_inputs

_log = logging.getLogger(__name__)

_FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "golf" / "fixtures"


@dataclass
class TournamentScan:
    tournament: str
    field_size: int
    sims: dict[str, SimResult]
    fields: list[FieldPrices]
    edges: list[GolfEdge]
    plays: list[GolfEdge]


@dataclass
class GolfScan:
    tournaments: list[TournamentScan] = dc_field(default_factory=list)

    @property
    def all_plays(self) -> list[GolfEdge]:
        out: list[GolfEdge] = []
        for t in self.tournaments:
            out.extend(t.plays)
        out.sort(key=lambda e: (e.edge_vs_raw or -1), reverse=True)
        return out


# ---------------------------------------------------------------------------
# Data loading (fixture or live)
# ---------------------------------------------------------------------------


def _load_skills(live: bool, year: int, cfg: SkillConfig):
    if live:
        from evmax.golf.data.pga import fetch_sg_snapshot

        snaps = fetch_sg_snapshot(year)
    else:
        snaps = parse_sg_snapshot(json.loads((_FIXTURES / "pga_sg_2026.json").read_text()))
    return skill_from_pga(snaps, cfg)


def _load_venue_fields(live: bool, include_kalshi: bool) -> list[FieldPrices]:
    fields: list[FieldPrices] = []
    if live:
        from evmax.golf.venues.polymarket_golf import fetch_golf as poly_fetch

        fields.extend(poly_fetch())
        if include_kalshi:
            try:
                from evmax.golf.venues.kalshi_golf import fetch_golf as kalshi_fetch

                fields.extend(kalshi_fetch())
            except Exception as exc:  # pragma: no cover - live/creds
                _log.warning("kalshi_golf_fetch_skipped err=%s", exc)
    else:
        from evmax.golf.venues.polymarket_golf import parse_golf_events
        from evmax.golf.venues.kalshi_golf import parse_kalshi_markets

        fields.extend(
            parse_golf_events(json.loads((_FIXTURES / "polyus_open_winner.json").read_text()))
        )
        if include_kalshi:
            fields.extend(
                parse_kalshi_markets(
                    json.loads((_FIXTURES / "kalshi_golf_markets.json").read_text())["markets"],
                    series_ticker="KXPGATOUR",
                )
            )
    return fields


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _tournament_key(name: str) -> str:
    """Collapse a market title to a tournament identity (drop the market family
    suffix so 'The Open Championship Winner' and '... Round 1 Leader' group)."""
    n = name.lower()
    for suffix in ("winner", "end of round 1 leader", "end of round 2 leader",
                   "end of round 3 leader", "round 1 leader", "round 2 leader",
                   "round 3 leader", "to win", "make the cut", "top 5", "top 10", "top 20"):
        n = n.replace(suffix, "")
    return " ".join(n.split()).strip(" -")


def scan(
    live: bool = False,
    include_kalshi: bool = True,
    year: int = 2026,
    sim_config: Optional[SimConfig] = None,
    skill_config: Optional[SkillConfig] = None,
    edge_config: Optional[EdgeConfig] = None,
) -> GolfScan:
    skill_config = skill_config or SkillConfig()
    edge_config = edge_config or EdgeConfig()

    skills = _load_skills(live, year, skill_config)
    venue_fields = _load_venue_fields(live, include_kalshi)

    # Group venue fields by tournament identity.
    groups: dict[str, list[FieldPrices]] = {}
    for f in venue_fields:
        groups.setdefault(_tournament_key(f.tournament), []).append(f)

    scan_result = GolfScan()
    for tkey, fields in groups.items():
        # The field to simulate: prefer a winner market's roster (full field),
        # else the union of every golfer quoted across this tournament.
        winner_fields = [f for f in fields if f.market_type is GolfMarketType.win]
        if winner_fields:
            roster = sorted({m.golfer for f in winner_fields for m in f.markets})
        else:
            roster = sorted({m.golfer for f in fields for m in f.markets})
        if len(roster) < 2:
            continue

        names, means, sds = to_sim_inputs(roster, skills, skill_config)
        # Cut size scales with field: majors cut to ~70, smaller fields less.
        sc = sim_config or SimConfig(cut_size=min(65, max(2, len(roster) // 2)))
        results = simulate_field(names, means, sds, sc)
        sims = {r.player: r for r in results}

        edges: list[GolfEdge] = []
        for f in fields:
            devig_field(f)
            edges.extend(price_field(sims, f, devig=False))
        plays = flag_plays(edges, fields, edge_config)

        title = winner_fields[0].tournament if winner_fields else fields[0].tournament
        scan_result.tournaments.append(
            TournamentScan(
                tournament=title,
                field_size=len(roster),
                sims=sims,
                fields=fields,
                edges=edges,
                plays=plays,
            )
        )
    return scan_result


# ---------------------------------------------------------------------------
# Rendering (Event + Outcome columns are mandatory per the CLI policy)
# ---------------------------------------------------------------------------


def _outcome_label(e: GolfEdge, display: dict[str, str]) -> str:
    name = display.get(e.golfer, e.golfer.title())
    label = {
        GolfMarketType.win: "Win",
        GolfMarketType.top_5: "Top 5",
        GolfMarketType.top_10: "Top 10",
        GolfMarketType.top_20: "Top 20",
        GolfMarketType.make_cut: "Make Cut",
    }.get(e.market_type, e.market_type.value)
    return f"{name} — {label}"


def render(scan_result: GolfScan, top: int = 25, display_names: Optional[dict] = None) -> str:
    display_names = display_names or {}
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console(record=True, width=120)
        for t in scan_result.tournaments:
            # One table per (venue, market family) — never mix venues in a table.
            by_group: dict[tuple, list[GolfEdge]] = {}
            for e in t.edges:
                by_group.setdefault((e.venue, e.market_type), []).append(e)
            for (venue, mt), edges in sorted(by_group.items(), key=lambda kv: (kv[0][0], kv[0][1].value)):
                fam = {GolfMarketType.win: "Winner", GolfMarketType.top_5: "Top 5",
                       GolfMarketType.top_10: "Top 10", GolfMarketType.top_20: "Top 20",
                       GolfMarketType.make_cut: "Make Cut"}.get(mt, mt.value)
                table = Table(
                    title=f"{t.tournament} · {venue} · {fam}  (field {t.field_size})",
                    show_lines=False,
                )
                table.add_column("Event", no_wrap=False, min_width=24)
                table.add_column("Outcome", no_wrap=False, min_width=18)
                table.add_column("Model", justify="right")
                table.add_column("Mkt(fair)", justify="right")
                table.add_column("Ask", justify="right")
                table.add_column("Edge", justify="right")
                table.add_column("Edge/Ask", justify="right")
                for e in sorted(edges, key=lambda e: e.model_prob, reverse=True)[:top]:
                    evr = e.edge_vs_raw
                    table.add_row(
                        t.tournament,
                        _outcome_label(e, display_names),
                        f"{e.model_prob:.1%}",
                        f"{e.market_prob:.1%}",
                        f"{e.market_raw:.1%}" if e.market_raw is not None else "-",
                        f"{e.edge:+.1%}",
                        f"{evr:+.1%}" if evr is not None else "-",
                    )
                console.print(table)
            if t.plays:
                console.print(f"[bold green]{len(t.plays)} flagged play(s)[/] "
                              f"(edge vs ask ≥ threshold, liquid)")
            else:
                console.print("[dim]no flagged plays (model does not beat the ask past threshold)[/]")
        return console.export_text()
    except ImportError:  # pragma: no cover
        lines = []
        for t in scan_result.tournaments:
            lines.append(f"== {t.tournament} (field {t.field_size}) ==")
            for e in sorted(t.edges, key=lambda e: e.model_prob, reverse=True)[:top]:
                lines.append(
                    f"{t.tournament} | {_outcome_label(e, display_names)} | "
                    f"model {e.model_prob:.1%} fair {e.market_prob:.1%} "
                    f"ask {e.market_raw:.1%} edge {e.edge:+.1%}"
                )
        return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI entry
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="evmax golf edge observation")
    ap.add_argument("--live", action="store_true", help="fetch live venues + PGA SG")
    ap.add_argument("--no-kalshi", action="store_true", help="skip Kalshi")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--sims", type=int, default=20000)
    args = ap.parse_args()

    result = scan(
        live=args.live,
        include_kalshi=not args.no_kalshi,
        year=args.year,
        sim_config=SimConfig(n_sims=args.sims),
    )
    # Display names from the PGA snapshot for nicer output.
    display = {}
    if not args.live:
        snaps = parse_sg_snapshot(json.loads((_FIXTURES / "pga_sg_2026.json").read_text()))
        display = {k: v.display_name for k, v in snaps.items()}
    print(render(result, display_names=display))


if __name__ == "__main__":  # pragma: no cover
    main()
