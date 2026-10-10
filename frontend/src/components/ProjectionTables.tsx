import { Fragment, useMemo, useState, type ReactNode } from 'react'
import type { ProjectionCell, ProjectionColumn, ProjectionGame, ProjectionPick, ProjectionPlayer, PickResult } from '../lib/types'

const pct = (p: number) => `${(p * 100).toFixed(0)}%`
function fmt(v: number | null | undefined, d = 0): string {
  if (v == null) return '—'
  const s = v.toFixed(d)
  return Number(s) === 0 ? (0).toFixed(d) : s // -0.3 rounds to "-0"; show "0"
}

function kickoffLabel(g: ProjectionGame): string {
  if (g.kickoff) {
    const t = new Date(g.kickoff)
    if (!isNaN(t.getTime())) {
      return t.toLocaleString(undefined, { weekday: 'short', month: 'numeric', day: 'numeric', hour: 'numeric', minute: '2-digit' })
    }
  }
  return g.date ?? '—'
}

/** A range cell (median with its low–high range) or a probability cell, plus the graded result once known. */
export function ProjectionCellView({ cell, kind }: { cell: ProjectionCell | null | undefined; kind: ProjectionColumn['kind'] }) {
  if (!cell) return <td className="num muted">—</td>
  return (
    <td className="num">
      {kind === 'prob' ? pct(cell.value) : fmt(cell.value)}
      {kind === 'range' && cell.lo != null && cell.hi != null && (
        <span className="muted proj-sub"> ({fmt(cell.lo)}–{fmt(cell.hi)})</span>
      )}
      {cell.sub && <span className="muted proj-sub"> · {cell.sub}</span>}
      {cell.result && (
        <div className={`proj-sub ${cell.hit === true ? 'proj-hit' : cell.hit === false ? 'proj-miss' : 'muted'}`}>
          {cell.result}
        </div>
      )}
    </td>
  )
}

/** W / L / P after a pick, graded at the line it was made against; the close goes in the tooltip. */
function ResultMark({ at, close }: { at: PickResult | null; close: PickResult | null }) {
  if (!at) return null
  const cls = at === 'W' ? 'green' : at === 'L' ? 'red' : 'muted'
  const title = `${at} at the line the pick was made against${close ? ` · ${close} at the closing line` : ''}`
  return <strong className={cls} title={title} style={{ marginLeft: 6 }}>{at}</strong>
}

/** The row's Outcome when the engine makes picks: the spread side and the over/under, with edges. */
function PickCell({ pick }: { pick: ProjectionPick | null | undefined }) {
  if (!pick) return <td className="muted">—</td>
  const vs = pick.line ? `Model pick vs ${pick.line}` : 'Model pick'
  return (
    <td style={{ whiteSpace: 'nowrap' }} title={pick.recorded ? `${vs} (recorded)` : vs}>
      {pick.spread
        ? <><strong style={{ color: 'var(--text)' }}>{pick.spread}</strong>
            <span className="muted proj-sub"> ({fmt(pick.spread_edge, 1)})</span>
            <ResultMark at={pick.spread_result} close={pick.spread_result_close} /></>
        : <span className="muted">—</span>}
      {pick.total && (
        <div className="proj-sub">
          {pick.total}<span className="muted"> ({fmt(pick.total_edge, 1)})</span>
          <ResultMark at={pick.total_result} close={pick.total_result_close} />
        </div>
      )}
    </td>
  )
}

interface GamesProps {
  games: ProjectionGame[]
  runLabel?: string
  /** game_id of the expanded row (its detail renders under it). */
  expanded: string | null
  onToggle?: (g: ProjectionGame) => void
  renderDetail: (g: ProjectionGame) => ReactNode
}

/**
 * One row per game: model line and total next to the market's, home win
 * probability, and the final score once graded. When the engine makes picks,
 * Outcome is the pick (spread side + over/under vs the market, edge in points,
 * W/L once graded) and the model's own line moves to a Model column. Each
 * row's action runs the engine's per-game model and expands the result in place.
 */
export function ProjectionGamesTable({ games, runLabel, expanded, onToggle, renderDetail }: GamesProps) {
  const hasPicks = games.some(g => g.pick)
  const cols = (onToggle ? 8 : 7) + (hasPicks ? 1 : 0)
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Kickoff</th>
            <th>Event</th>
            {hasPicks && (
              <th title="Model pick vs the market line: the spread side and over/under the model prefers, edge in points; W/L once graded">
                Outcome
              </th>
            )}
            <th title="Model line (favorite) and projected total">{hasPicks ? 'Model' : 'Outcome'}</th>
            <th className="num" title="Projected score, away – home">Score</th>
            <th className="num" title="Model home win probability">Home win</th>
            <th title="The market's line and total — comparison only, never a model input">Market</th>
            <th className="num">Final</th>
            {onToggle && <th aria-label="Per-game model" />}
          </tr>
        </thead>
        <tbody>
          {games.map(g => {
            const open = expanded === g.game_id
            return (
              <Fragment key={g.game_id}>
                <tr className={open ? 'proj-row-open' : undefined}>
                  <td className="muted" style={{ whiteSpace: 'nowrap' }}>{kickoffLabel(g)}</td>
                  <td>
                    <span style={{ color: 'var(--text)' }}>{g.away_name} @ {g.home_name}</span>
                    {g.neutral && <span className="muted"> (neutral)</span>}
                    {g.flags.map(f => <span key={f} className="badge warn proj-flag">{f}</span>)}
                    {g.subtitle && <div className="muted proj-sub">{g.subtitle}</div>}
                  </td>
                  {hasPicks && <PickCell pick={g.pick} />}
                  <td>{g.model_line ?? '—'} · total {(g.pick?.model_total ?? g.proj_total).toFixed(1)}</td>
                  <td className="num" style={{ whiteSpace: 'nowrap' }}>{g.proj_away.toFixed(1)} – {g.proj_home.toFixed(1)}</td>
                  <td className="num">{pct(g.p_home_win)}</td>
                  <td className="muted">
                    {g.market_line == null && g.market_total == null
                      ? '—'
                      : `${g.market_line ?? '—'} · ${g.market_total != null ? g.market_total.toFixed(1) : '—'}`}
                  </td>
                  <td className="num" style={{ whiteSpace: 'nowrap' }}>
                    {g.actual_home == null || g.actual_away == null ? '—' : `${g.actual_away} – ${g.actual_home}`}
                  </td>
                  {onToggle && (
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className={`btn btn-sm ${open ? 'success' : ''}`}
                        aria-expanded={open}
                        onClick={() => onToggle(g)}
                      >
                        {open ? 'Hide' : (runLabel ?? 'Run')}
                      </button>
                    </td>
                  )}
                </tr>
                {open && (
                  <tr className="proj-detail-row">
                    <td colSpan={cols}>{renderDetail(g)}</td>
                  </tr>
                )}
              </Fragment>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

interface PlayersProps {
  rows: ProjectionPlayer[]
  columns: ProjectionColumn[]
  /** Show the team filter + sort controls (the slate view); off inside a game's detail. */
  controls?: boolean
  defaultSort?: string | null
  /** Show the Event column (off when the table is already scoped to one game). */
  showEvent?: boolean
}

/** Player rows rendered from the engine's column specs; sorting by a column hides rows without that stat. */
export function ProjectionPlayerTable({ rows, columns, controls = false, defaultSort, showEvent = true }: PlayersProps) {
  const [team, setTeam] = useState('')
  const [sortKey, setSortKey] = useState<string>(defaultSort ?? columns[0]?.key ?? '')
  const teams = useMemo(() => Array.from(new Set(rows.map(r => r.team))).sort(), [rows])

  const shown = useMemo(() => {
    if (!controls) return rows
    const key = columns.some(c => c.key === sortKey) ? sortKey : columns[0]?.key
    return rows
      .filter(r => (!team || r.team === team) && (!key || r.cells[key] != null))
      .sort((a, b) => (b.cells[key]?.value ?? 0) - (a.cells[key]?.value ?? 0))
  }, [rows, columns, controls, team, sortKey])

  return (
    <>
      {controls && (
        <div className="proj-controls" style={{ marginBottom: 12 }}>
          <select value={team} onChange={e => setTeam(e.target.value)} aria-label="Team">
            <option value="">All teams</option>
            {teams.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
          <select value={sortKey} onChange={e => setSortKey(e.target.value)} aria-label="Sort by">
            {columns.map(c => <option key={c.key} value={c.key}>Sort: {c.label}</option>)}
          </select>
          <span className="muted" style={{ fontSize: 12 }}>{shown.length} players</span>
        </div>
      )}
      <div className={controls ? 'scroll-area proj-players-scroll' : 'table-scroll'}>
        <table>
          <thead>
            <tr>
              {showEvent && <th>Event</th>}
              <th>Outcome</th>
              {columns.map(c => <th key={c.key} className="num" title={c.title || undefined}>{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {shown.map(r => (
              <tr key={`${r.game_id}-${r.player_id}`} style={r.dimmed ? { opacity: 0.5 } : undefined}>
                {showEvent && <td className="muted">{r.event}</td>}
                <td>
                  <span style={{ color: 'var(--text)' }}>{r.name}</span> <span className="muted">({r.detail})</span>
                  {r.note && <div className="muted proj-sub">{r.note}</div>}
                </td>
                {columns.map(c => <ProjectionCellView key={c.key} cell={r.cells[c.key]} kind={c.kind} />)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
