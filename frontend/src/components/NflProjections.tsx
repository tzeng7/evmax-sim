import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchNflProjections } from '../lib/api'
import type { NflPlayerProjection, NflProjectionsResult } from '../lib/types'

interface Props {
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
}

type StatKey = 'receiving_yards' | 'receptions' | 'rushing_yards' | 'passing_yards'
type SortKey = StatKey | 'anytime_td'

const STAT_LABEL: Record<StatKey, string> = {
  receiving_yards: 'Rec yds',
  receptions: 'Rec',
  rushing_yards: 'Rush yds',
  passing_yards: 'Pass yds',
}
const SORT_LABEL: Record<SortKey, string> = { ...STAT_LABEL, anytime_td: 'Anytime TD' }

/** P(anytime TD) with the result once graded; starting QBs also show projected passing TDs. */
function TdCell({ p }: { p: NflPlayerProjection }) {
  if (p.p_anytime_td == null) return <td style={{ textAlign: 'right' }} className="muted">—</td>
  return (
    <td style={{ textAlign: 'right' }}>
      {(p.p_anytime_td * 100).toFixed(0)}%
      {!!p.is_starting_qb && p.proj_passing_tds != null && (
        <span className="muted" style={{ fontSize: 11 }}> · {p.proj_passing_tds.toFixed(1)} pass</span>
      )}
      {p.actual_tds != null && (
        <div style={{ fontSize: 11, color: p.actual_tds > 0 ? 'var(--green)' : 'var(--muted)' }}>
          {p.actual_tds > 0 ? `scored ${p.actual_tds}` : 'no TD'}
        </div>
      )}
    </td>
  )
}

const fmt = (v: number | null | undefined, d = 0) => (v == null ? '—' : v.toFixed(d))

function kickoff(utc: string | null, gameday: string): string {
  if (!utc) return gameday
  const t = new Date(utc)
  return t.toLocaleString(undefined, { weekday: 'short', hour: 'numeric', minute: '2-digit' })
}

/** Median with the 10th–90th percentile range, plus the result once graded. */
function StatCell({ p, stat }: { p: NflPlayerProjection; stat: StatKey }) {
  const med = p[`proj_${stat}`]
  const lo = p[`p10_${stat}`]
  const hi = p[`p90_${stat}`]
  const actual = p[`actual_${stat}`]
  const shown = stat === 'passing_yards' ? !!p.is_starting_qb
    : stat === 'rushing_yards' ? (p.proj_carries ?? 0) >= 1
    : (p.proj_targets ?? 0) >= 1
  if (!shown || med == null) return <td style={{ textAlign: 'right' }} className="muted">—</td>
  const inRange = actual != null && lo != null && hi != null && actual >= lo && actual <= hi
  return (
    <td style={{ textAlign: 'right' }}>
      {fmt(med)} <span className="muted" style={{ fontSize: 11 }}>({fmt(lo)}–{fmt(hi)})</span>
      {actual != null && (
        <div style={{ fontSize: 11, color: inRange ? 'var(--green)' : 'var(--amber)' }}>actual {fmt(actual)}</div>
      )}
    </td>
  )
}

/**
 * NFL projections — the standalone evmax.nfl_projections model (no market
 * inputs): game scores / line / total next to the market's consensus line, and
 * player medians with 10th–90th percentile ranges. Rows are written by the
 * scheduled `evmax project nfl-run` and graded by `nfl-resolve`.
 */
export function NflProjections({ toast }: Props) {
  const [sel, setSel] = useState<{ season?: number; week?: number }>({})
  const [res, setRes] = useState<NflProjectionsResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [team, setTeam] = useState('')
  const [sortStat, setSortStat] = useState<SortKey>('receiving_yards')

  const load = useCallback(async (season?: number, week?: number) => {
    setLoading(true)
    try {
      setRes(await fetchNflProjections(season, week))
    } catch (e) {
      toast('NFL projections load failed: ' + (e as Error).message, 'err')
    } finally {
      setLoading(false)
    }
  }, [toast])

  useEffect(() => { load(sel.season, sel.week) }, [sel, load])

  const teams = useMemo(
    () => Array.from(new Set((res?.players ?? []).map(p => p.team))).sort(),
    [res],
  )

  const players = useMemo(() => {
    const ps = (res?.players ?? []).filter(p => {
      if (team && p.team !== team) return false
      if (sortStat === 'passing_yards') return !!p.is_starting_qb
      if (sortStat === 'rushing_yards') return (p.proj_carries ?? 0) >= 3
      if (sortStat === 'anytime_td') return p.p_anytime_td != null
      return (p.proj_targets ?? 0) >= 2
    })
    const key = (p: NflPlayerProjection) => (sortStat === 'anytime_td' ? p.p_anytime_td : p[`proj_${sortStat}`]) ?? 0
    return ps.sort((a, b) => key(b) - key(a))
  }, [res, team, sortStat])

  const acc = res?.accuracy
  const weekKey = res?.season != null ? `${res.season}-${res.week}` : ''

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
        <h2 style={{ margin: 0 }}>NFL Projections</h2>
        <span className="muted">Model only (no market inputs) · medians with 10th–90th percentile ranges</span>
        <div style={{ flex: 1 }} />
        {res && res.weeks.length > 0 && (
          <select
            value={weekKey}
            onChange={e => {
              const [s, w] = e.target.value.split('-').map(Number)
              setSel({ season: s, week: w })
            }}
          >
            {res.weeks.map(w => (
              <option key={`${w.season}-${w.week}`} value={`${w.season}-${w.week}`}>
                {w.season} · Week {w.week}
              </option>
            ))}
          </select>
        )}
      </div>

      {loading && <p className="muted">Loading…</p>}
      {!loading && res && res.games.length === 0 && (
        <p className="muted">
          No NFL projections stored yet. Run <code>evmax project nfl-run</code> (the scheduled weekly task does this).
        </p>
      )}

      {!loading && res && res.games.length > 0 && (
        <>
          {acc && acc.games.n > 0 && (
            <p className="muted" style={{ marginTop: 0 }}>
              {res.season} tracked: margin MAE {fmt(acc.games.margin_mae, 2)} (closing line {fmt(acc.games.close_margin_mae, 2)}),
              total MAE {fmt(acc.games.total_mae, 2)} (closing line {fmt(acc.games.close_total_mae, 2)}) over {acc.games.n} games
              {Object.entries(acc.players).filter(([k]) => k in STAT_LABEL).map(([k, m]) => (
                <span key={k}> · {STAT_LABEL[k as StatKey]} MAE {fmt(m.mae, 1)} (n {m.n})</span>
              ))}
              {acc.players.anytime_td && (
                <span> · anytime TD: predicted {fmt((acc.players.anytime_td.mean_p ?? 0) * 100, 0)}% vs actual{' '}
                  {fmt((acc.players.anytime_td.rate ?? 0) * 100, 0)}% (Brier {fmt(acc.players.anytime_td.brier, 3)})</span>
              )}
            </p>
          )}

          <table className="bets-table" style={{ width: '100%', marginBottom: 24 }}>
            <thead>
              <tr>
                <th>Kickoff</th>
                <th>Event</th>
                <th>Outcome</th>
                <th style={{ textAlign: 'right' }}>Score</th>
                <th style={{ textAlign: 'right' }} title="Model home win probability">Home win</th>
                <th title="nflverse consensus line at the last run — comparison only, never a model input">Market</th>
                <th style={{ textAlign: 'right' }}>Final</th>
              </tr>
            </thead>
            <tbody>
              {res.games.map(g => (
                <tr key={g.game_id}>
                  <td className="muted">{kickoff(g.kickoff_utc, g.gameday)}</td>
                  <td>
                    {g.away_team} @ {g.home_team}{g.neutral ? ' (neutral)' : ''}
                    <div className="muted" style={{ fontSize: 11 }}>{g.away_qb_name ?? '?'} / {g.home_qb_name ?? '?'}</div>
                  </td>
                  <td>{g.model_line} · total {g.proj_total.toFixed(1)}</td>
                  <td style={{ textAlign: 'right' }}>{g.away_team} {g.proj_away.toFixed(1)} – {g.home_team} {g.proj_home.toFixed(1)}</td>
                  <td style={{ textAlign: 'right' }}>{(g.p_home_win * 100).toFixed(0)}%</td>
                  <td className="muted">{g.market_home_margin == null ? '—' : `${g.market_line} · ${fmt(g.market_total, 1)}`}</td>
                  <td style={{ textAlign: 'right' }}>
                    {g.actual_home == null ? '—' : `${g.away_team} ${g.actual_away} – ${g.home_team} ${g.actual_home}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
            <h3 style={{ margin: 0 }}>Players</h3>
            <select value={team} onChange={e => setTeam(e.target.value)}>
              <option value="">All teams</option>
              {teams.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
            <select value={sortStat} onChange={e => setSortStat(e.target.value as SortKey)}>
              {(Object.keys(SORT_LABEL) as SortKey[]).map(k => <option key={k} value={k}>Sort: {SORT_LABEL[k]}</option>)}
            </select>
            <span className="muted">{players.length} players</span>
          </div>
          <table className="bets-table" style={{ width: '100%' }}>
            <thead>
              <tr>
                <th>Event</th>
                <th>Outcome</th>
                <th style={{ textAlign: 'right' }}>Rec</th>
                <th style={{ textAlign: 'right' }}>Rec yds</th>
                <th style={{ textAlign: 'right' }}>Rush yds</th>
                <th style={{ textAlign: 'right' }}>Pass yds</th>
                <th style={{ textAlign: 'right' }} title="Probability of a rushing or receiving touchdown (Poisson on expected TDs)">Anytime TD</th>
              </tr>
            </thead>
            <tbody>
              {players.map(p => (
                <tr key={`${p.game_id}-${p.player_id}`} style={p.played === 0 ? { opacity: 0.5 } : undefined}>
                  <td className="muted">{p.team} vs {p.opp}</td>
                  <td>
                    {p.player_name ?? p.player_id} <span className="muted">({p.position}, {p.team})</span>
                    {p.played === 0 && <div className="muted" style={{ fontSize: 11 }}>did not play</div>}
                  </td>
                  <StatCell p={p} stat="receptions" />
                  <StatCell p={p} stat="receiving_yards" />
                  <StatCell p={p} stat="rushing_yards" />
                  <StatCell p={p} stat="passing_yards" />
                  <TdCell p={p} />
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      <p className="muted" style={{ fontSize: 12, marginTop: 12 }}>
        Walk-forward 2020–25: margin MAE 10.13 (Vegas close 9.76), total MAE 10.49 (10.28). Player medians beat a
        last-8-games average by 6–11% on the 2025 holdout but trail the Kalshi market by 3–6%. Teammates of players
        ruled out absorb 60% of their targets and carries. Ranges: about 10% of results should fall below the low end
        and 10% above the high end.
      </p>
    </>
  )
}
