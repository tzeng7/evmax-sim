"""Tests for the canonical evaluation (evmax.agents.cleanup.evaluation).

Pins the funnel stage order, own-side ROI math, gross-vs-net fees, game
clustering and byte-for-byte determinism. Runs with max_staleness_h=None so
archive.db is never touched.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from evmax.agents.cleanup import evaluation as ev
from evmax.fees import venue_fee_prob

GAME_DATE = "2026-10-01"


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE ev_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at TEXT, scan_date TEXT, market_id TEXT UNIQUE, event_id TEXT,
            sector TEXT, yes_team TEXT, market_type TEXT, event_date TEXT,
            kalshi_yes_price REAL, sharp_true_prob REAL, blended_true_prob REAL,
            model_sources TEXT, line REAL,
            voided INTEGER NOT NULL DEFAULT 0, void_reason TEXT,
            placed INTEGER NOT NULL DEFAULT 0, placed_at TEXT, placed_price REAL,
            mode TEXT NOT NULL DEFAULT 'live', venue TEXT NOT NULL DEFAULT 'kalshi',
            kalshi_clv_pct REAL, minutes_to_tipoff INTEGER, league TEXT
        );
        CREATE TABLE ev_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, market_id TEXT UNIQUE,
            outcome INTEGER, resolved_at TEXT
        );
        """
    )
    return conn


def _ins(conn, mid, outcome=1, price=0.5, sector="wnba", mt="moneyline",
         sources="elo+sharp", clv=1.0, voided=0, void_reason=None, mtt=120,
         event_id=None, venue="kalshi", line=None, blended=0.6, sharp=0.55):
    conn.execute(
        """INSERT INTO ev_predictions (logged_at, scan_date, market_id, event_id,
           sector, yes_team, market_type, event_date, kalshi_yes_price,
           sharp_true_prob, blended_true_prob, model_sources, line, voided,
           void_reason, mode, venue, kalshi_clv_pct, minutes_to_tipoff)
           VALUES ('2026-10-01 10:00:00',?,?,?,?,'t',?,?,?,?,?,?,?,?,?,'shadow',?,?,?)""",
        (GAME_DATE, mid, event_id or f"{sector}::{GAME_DATE}::{mid}", sector, mt,
         GAME_DATE, price, sharp, blended, sources, line, voided, void_reason,
         venue, clv, mtt),
    )
    if outcome is not None:
        conn.execute(
            "INSERT INTO ev_outcomes (market_id, outcome, resolved_at) VALUES (?,?,?)",
            (mid, outcome, "2026-10-02 00:00:00"),
        )
    conn.commit()


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "predictions.db"
    conn = _make_db(path)
    from evmax.agents.cleanup import db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", path)
    return conn


def _spec(**kw):
    kw.setdefault("max_staleness_h", None)
    return ev.EvalSpec(**kw)


def _group(spec):
    return ev.run_eval(spec)["groups"][0]


def _funnel(g):
    return {f["stage"]: f["n_rows"] for f in g["funnel"]}


def test_funnel_drops_each_stage_in_order(db):
    _ins(db, "ok")
    _ins(db, "unresolved", outcome=None)
    _ins(db, "manual_void", voided=1, void_reason="manual")
    _ins(db, "stale_reverted", voided=1, void_reason="stale_reverted")
    _ins(db, "inplay", mtt=0)
    _ins(db, "bad_price", price=0.0)
    _ins(db, "contaminated", sector="baseball", sources="elo+sharp")  # no pitcher
    _ins(db, "partial", sector="tennis", sources="tennis_surface+sharp")  # not live-eligible
    g = _group(_spec())
    assert [f["stage"] for f in g["funnel"]][:7] == [
        "scope", "resolved", "not_voided", "pre_game_entry",
        "valid_entry_price", "current_code", "live_eligible",
    ]
    f = _funnel(g)
    assert f["scope"] == 8
    assert f["resolved"] == 7
    assert f["not_voided"] == 6          # manual void dropped, stale_reverted kept
    assert f["pre_game_entry"] == 5
    assert f["valid_entry_price"] == 4
    assert f["current_code"] == 3        # baseball no-pitcher contaminated
    assert f["live_eligible"] == 2       # tennis partial blend demoted


def test_population_clean_keeps_live_ineligible(db):
    _ins(db, "ok")
    _ins(db, "partial", sector="tennis", sources="tennis_surface+sharp")
    assert _group(_spec(population="gate"))["roi_population"]["n_rows"] == 1
    assert _group(_spec(population="clean"))["roi_population"]["n_rows"] == 2


def test_roi_gross_own_side_math(db):
    _ins(db, "w", outcome=1, price=0.5)
    _ins(db, "l", outcome=0, price=0.5)
    r = _group(_spec())["roi_population"]
    assert r["roi_gross_pct"] == 0.0  # +1.0 and -1.0 on $1 notional at 50c


def test_no_side_row_uses_same_formula(db):
    # NBA spreads are layered since 2026-11-16: only a rung priced off Pinnacle's
    # own rung (`sharp_ladder`) is live-eligible, so the row must carry it to be
    # in the ROI population.
    _ins(db, "kalshi:KXNBASPREAD-X-A:no", outcome=1, price=0.8, mt="spread", sector="nba",
         line=-5.5, sources="sharp+sharp_ladder+no_side")
    r = _group(_spec())["roi_population"]
    assert r["roi_gross_pct"] == pytest.approx(25.0)  # 1/0.8 - 1


def test_net_roi_is_below_gross_by_fee(db):
    _ins(db, "w", outcome=1, price=0.5)
    r = _group(_spec())["roi_population"]
    assert r["roi_net_pct"] < r["roi_gross_pct"]


def test_n_games_clusters_ladder_rungs(db):
    for i in range(3):
        _ins(db, f"rung{i}", sector="nfl", mt="spread",
             event_id=f"nfl::{GAME_DATE}::a_vs_b::spread::{i}.5", line=-i - 0.5,
             sources="sharp_ladder")
    r = _group(_spec(sector="nfl"))["roi_population"]
    assert r["n_rows"] == 3 and r["n_games"] == 1


def test_clv_net_subtracts_entry_fee(db):
    _ins(db, "a", clv=2.0, price=0.5)
    c = _group(_spec())["clv_population"]
    assert c["gross"]["mean_pp"] == 2.0
    assert c["net_of_fees"]["mean_pp"] == pytest.approx(
        2.0 - 100 * venue_fee_prob("kalshi", 0.5), abs=1e-3
    )


def test_clv_population_excludes_null_clv_but_roi_keeps_it(db):
    _ins(db, "a", clv=None)
    _ins(db, "b", clv=1.0)
    g = _group(_spec())
    assert g["roi_population"]["n_rows"] == 2
    assert g["clv_population"]["n_rows"] == 1
    assert _funnel(g)["clv_present"] == 1


def test_deterministic_and_digest_tracks_rows(db):
    for i in range(6):
        _ins(db, f"m{i}", outcome=i % 2, price=0.4)
    a = ev.to_json(ev.run_eval(_spec()))
    b = ev.to_json(ev.run_eval(_spec()))
    assert a == b
    d1 = _group(_spec())["roi_population"]["row_digest"]
    _ins(db, "extra")
    assert _group(_spec())["roi_population"]["row_digest"] != d1


def test_group_by_splits_and_filters_apply(db):
    _ins(db, "a", sector="wnba")
    _ins(db, "b", sector="ncaab")
    _ins(db, "c", sector="ncaab", venue="polymarket_us")
    res = ev.run_eval(_spec(group_by=("sector", "venue")))
    keys = {(g["group"]["sector"], g["group"]["venue"]) for g in res["groups"]}
    assert keys == {("wnba", "kalshi"), ("ncaab", "kalshi"), ("ncaab", "polymarket_us")}
    only = ev.run_eval(_spec(sector="ncaab", venue="kalshi"))["groups"][0]
    assert only["roi_population"]["n_rows"] == 1


def test_event_date_window(db):
    _ins(db, "a")
    assert _group(_spec(since="2026-10-02"))["roi_population"]["n_rows"] == 0
    assert _group(_spec(until="2026-10-01"))["roi_population"]["n_rows"] == 1


def test_empty_population_does_not_crash(db):
    g = _group(_spec(sector="nope"))
    assert g["roi_population"]["n_rows"] == 0
    assert g["clv_population"]["clears_net"] is False


def test_invalid_spec_rejected():
    with pytest.raises(ValueError):
        ev.EvalSpec(population="bogus")
    with pytest.raises(ValueError):
        ev.EvalSpec(group_by=("league",))
    with pytest.raises(ValueError):
        ev.EvalSpec(side="both")


# --- coverage / realized dollars -------------------------------------------

def _place(conn, mid, stake, price, maker=0):
    conn.execute(
        "UPDATE ev_predictions SET placed=1, placed_stake=?, placed_price=? WHERE market_id=?",
        (stake, price, mid),
    )
    if maker:
        conn.execute("UPDATE ev_predictions SET maker_fill=1 WHERE market_id=?", (mid,))
    conn.commit()


@pytest.fixture
def db_pl(db):
    db.execute("ALTER TABLE ev_predictions ADD COLUMN placed_stake REAL")
    db.execute("ALTER TABLE ev_predictions ADD COLUMN maker_fill INTEGER NOT NULL DEFAULT 0")
    db.commit()
    return db


def test_empty_population_names_the_stage(db):
    _ins(db, "pending", outcome=None)
    g = _group(_spec())
    assert "none resolved" in g["roi_empty_reason"]
    _ins(db, "v", voided=1, void_reason="manual")
    g = _group(_spec(sector="wnba"))
    assert g["roi_empty_reason"] is not None


def test_clv_uncapped_sensitivity_reported_when_cap_set(db, monkeypatch):
    _ins(db, "a", clv=1.0)
    _ins(db, "b", clv=1.0)
    # cap drops everything (no archive snapshots in the test DB)
    monkeypatch.setattr(ev, "_fresh_close", lambda rows, h: ([], len(rows)))
    g = _group(ev.EvalSpec(max_staleness_h=3.0))
    assert g["clv_population"]["n_rows"] == 0
    assert g["clv_population"]["uncapped"]["n_rows"] == 2
    assert "stale or missing" in g["clv_population"]["empty_reason"]
    assert _group(_spec())["clv_population"]["uncapped"] is None


def test_realized_uses_real_stake_and_brackets_fee_treatment(db_pl):
    _ins(db_pl, "w", outcome=1, price=0.5)
    _ins(db_pl, "l", outcome=0, price=0.5)
    _place(db_pl, "w", 10.0, 0.40)   # round cent => raw fill, fee added
    _place(db_pl, "l", 10.0, 0.40)
    r = _group(_spec())["realized"]
    assert r["n_bets"] == 2 and r["staked_usd"] == 20.0
    # gross: +15 on win, -10 on loss = +5 ; raw includes fees so is lower
    assert r["pnl_fee_inclusive_usd"] == pytest.approx(5.0)
    assert r["pnl_raw_usd"] < r["pnl_fee_inclusive_usd"]
    assert r["pnl_est_usd"] == pytest.approx(r["pnl_raw_usd"])
    assert r["price_kinds"] == {"round_cent_raw": 2, "decimal_odds_fee_inclusive": 0,
                                "unknown_treated_raw": 0}


def test_realized_odds_derived_price_is_treated_fee_inclusive(db_pl):
    _ins(db_pl, "w", outcome=1, price=0.5)
    _place(db_pl, "w", 10.0, 1 / 2.05)   # dashboard fill: 2dp decimal odds
    r = _group(_spec())["realized"]
    assert r["price_kinds"] == {"round_cent_raw": 0, "decimal_odds_fee_inclusive": 1,
                                "unknown_treated_raw": 0}
    assert r["pnl_est_usd"] == pytest.approx(r["pnl_fee_inclusive_usd"])


def test_realized_ignores_unplaced_unresolved_and_cancelled(db_pl):
    _ins(db_pl, "unplaced")
    _ins(db_pl, "pending", outcome=None)
    _ins(db_pl, "cancelled", voided=1, void_reason="manual")
    for m in ("pending", "cancelled"):
        _place(db_pl, m, 5.0, 0.4)
    r = _group(_spec())["realized"]
    assert r["n_bets"] == 0 and "no placed bets" in r["empty_reason"]


# --- uniform structure across sectors --------------------------------------

def _shape(x, p=""):
    out = set()
    if isinstance(x, dict):
        for k, v in x.items():
            out.add(p + k)
            out |= _shape(v, p + k + ".")
    return out


def test_every_group_has_identical_schema_and_stages(db):
    """Sector, size, emptiness and population must never change output shape."""
    _ins(db, "w1", sector="wnba")
    _ins(db, "n1", sector="nba", clv=None)
    _ins(db, "t1", sector="tennis", sources="tennis_surface+sharp")
    _ins(db, "p1", sector="nfl", outcome=None)
    _ins(db, "v1", sector="ufc", voided=1, void_reason="manual")
    for pop in ("gate", "clean", "raw"):
        groups = ev.run_eval(_spec(group_by=("sector",), population=pop))["groups"]
        assert len(groups) == 5
        shapes = {frozenset(_shape({k: v for k, v in g.items() if k != "group"}))
                  for g in groups}
        stages = {tuple(f["stage"] for f in g["funnel"]) for g in groups}
        assert len(shapes) == 1, "output keys differ between sectors"
        assert len(stages) == 1, "funnel stages differ between sectors"


def test_evaluation_module_has_no_sector_branches():
    src = Path(ev.__file__).read_text()
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith(("#", '"""'))
    )
    for name in ("'nba'", '"nba"', '"wnba"', '"nfl"', '"soccer"', '"tennis"',
                 '"baseball"', '"ncaaf"', '"nhl"', '"ufc"'):
        assert name not in code, f"sector literal {name} in evaluation.py"


def test_conform_rejects_unknown_fields():
    with pytest.raises(KeyError):
        ev.conform({"surprise": 1}, ev.ROI_TEMPLATE)


def test_sector_rules_reported_with_same_keys_for_every_sector(db):
    _ins(db, "a", sector="wnba")
    _ins(db, "b", sector="soccer", sources="elo+sharp")
    res = {r["sector"]: r for r in ev.run_eval(_spec())["groups"][0]["sector_rules"]}
    assert set(res) == {"wnba", "soccer"}
    assert {frozenset(v) for v in res.values()} == {
        frozenset({"sector", "contamination_rules", "required_blend_models",
                   "min_nonsharp_models"})
    }
    assert res["soccer"]["min_nonsharp_models"]["min_count"] == 1
    assert res["wnba"]["min_nonsharp_models"] is None


def test_staleness_cap_applies_to_every_venue(db, monkeypatch):
    """Venue must not exempt a row from the cap (it used to exempt PolyUS)."""
    import evmax.archiver as archiver_mod

    seen = []

    class FakeArchiver:
        def get_kalshi_close_staleness_h(self, ticker, event_id, **kw):
            seen.append(ticker)
            return {"kfresh": 1.0, "kstale": 9.0, "pfresh": 0.5, "pstale": 9.0}.get(
                ticker.split(":")[-1].lower(), None
            )

    monkeypatch.setattr(archiver_mod, "DataArchiver", FakeArchiver)
    _ins(db, "kalshi:KFRESH", venue="kalshi")
    _ins(db, "kalshi:KSTALE", venue="kalshi")
    _ins(db, "polymarket_us:pfresh", venue="polymarket_us")
    _ins(db, "polymarket_us:pstale", venue="polymarket_us")
    _ins(db, "polymarket_us:pmissing", venue="polymarket_us")
    c = _group(ev.EvalSpec(max_staleness_h=3.0))["clv_population"]
    assert c["n_rows"] == 2                      # kfresh + pfresh only
    assert c["uncapped"]["n_rows"] == 5
    assert set(seen) == {"KFRESH", "KSTALE", "polymarket_us:pfresh",
                         "polymarket_us:pstale", "polymarket_us:pmissing"}
