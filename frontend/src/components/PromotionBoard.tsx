import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchPromotionBoard } from '../lib/api'
import { ModesPanel } from './ModesPanel'
import type { BoardRow } from '../lib/types'

interface Props {
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
}

// One entry per verdict kind promotion_board.VERDICT_KINDS can emit
// (tests/test_promotion_board.py fails if a kind is missing here).
// COLLECTING rows carry an "n/30[g]" suffix; verdictKind() strips it.
type VerdictKind =
  | 'SHARP-PASSTHROUGH' | 'COLLECTING' | 'FAILING-CLV'
  | 'PROMOTE-READY' | 'LIVE-DEGRADING' | 'LIVE-HEALTHY'

const VERDICT_STYLE: Record<VerdictKind, { color: string; legend: string }> = {
  'SHARP-PASSTHROUGH': { color: 'var(--red)', legend: 'Moneyline blend within 0.5pp of sharp: no independent model signal' },
  'LIVE-DEGRADING': { color: 'var(--red)', legend: 'Live lane with negative CLV over enough games' },
  'FAILING-CLV': { color: 'var(--amber)', legend: 'Shadow lane with enough games but CLV gate not met' },
  'COLLECTING': { color: 'var(--muted)', legend: 'Not enough clean resolved rows / CLV games yet (n/30)' },
  'PROMOTE-READY': { color: 'var(--green)', legend: 'Shadow lane clears every gate: promote by hand (CLI/YAML)' },
  'LIVE-HEALTHY': { color: 'var(--green)', legend: 'Live lane, no degradation signal' },
}

const verdictKind = (v: string): VerdictKind =>
  (v.startsWith('COLLECTING') ? 'COLLECTING' : v) as VerdictKind
const verdictColor = (v: string) => VERDICT_STYLE[verdictKind(v)]?.color ?? 'var(--muted)'

const PRICE_BUCKETS = ['0-10', '10-20', '20-35', '35-50', '50-65', '65-80', '80-90', '90+']
const SOCCER_LEAGUES = [
  'epl', 'laliga', 'bundesliga', 'seriea', 'ligue1', 'ucl', 'uel', 'mls',
  'ligamx', 'jleague', 'eredivisie', 'brasileirao', 'championship',
]
const GATE_LABEL: Record<string, string> = {
  clean_n: 'Clean live-eligible resolved rows',
  clv_n: 'CLV sample (independent games)',
  clv_mean: 'Mean CLV (pp)',
  clv_frac_pos: 'Fraction of games with positive CLV',
}

const MKT_ABBR: Record<string, string> = {
  moneyline: 'ML',
  spread: 'Spread',
  total: 'Total',
  advance: 'Advance',
}

/**
 * Promotion board — per (sector, market type, venue) health. The dashboard
 * face of `evmax cleanup shadow board`: which sectors can be relied on,
 * which are collecting sample, and which blends are sharp-passthrough
 * (divergence < 0.5pp = no independent model signal, greyed out).
 */
export function PromotionBoard({ toast }: Props) {
  const [days, setDays] = useState(30)
  const [rows, setRows] = useState<BoardRow[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [sector, setSector] = useState('')
  const [league, setLeague] = useState('')
  const [bucket, setBucket] = useState('')
  const [venue, setVenue] = useState('')
  const [knownSectors, setKnownSectors] = useState<string[]>([])
  const [open, setOpen] = useState<BoardRow | null>(null)

  const load = useCallback(async (d: number, sec: string, lg: string, bk: string) => {
    setLoading(true)
    try {
      const res = await fetchPromotionBoard(d, {
        sector: sec || undefined,
        league: lg || undefined,
        priceBucket: bk || undefined,
      })
      setRows(res.rows)
      setKnownSectors(prev => Array.from(new Set([...prev, ...res.rows.map(r => r.sector)])).sort())
    } catch (e) {
      toast('Board load failed: ' + (e as Error).message, 'err')
    } finally {
      setLoading(false)
    }
  }, [toast])

  useEffect(() => { load(days, sector, league, bucket) }, [days, sector, league, bucket, load])

  // Venue has no API param (rows already carry it): filter client-side.
  const shown = useMemo(
    () => (rows ?? []).filter(r => !venue || r.venue === venue),
    [rows, venue],
  )

  const copy = (text: string) =>
    navigator.clipboard.writeText(text).then(
      () => toast('Copied', 'ok'),
      () => toast('Copy failed', 'err'),
    )

  const gateOrder = ['clean_n', 'clv_n', 'clv_mean', 'clv_frac_pos']

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
        <h2 style={{ margin: 0 }}>Promotion Board</h2>
        <span className="muted">
          Blend divergence vs sharp + CLV promotion gates, per sector · market · venue
        </span>
        <div style={{ flex: 1 }} />
        <select value={sector} onChange={e => setSector(e.target.value)} aria-label="Sector">
          <option value="">All sectors</option>
          {knownSectors.map(x => <option key={x} value={x}>{x}</option>)}
        </select>
        <select value={venue} onChange={e => setVenue(e.target.value)} aria-label="Venue">
          <option value="">All venues</option>
          <option value="kalshi">Kalshi</option>
          <option value="polymarket_us">PolyUS</option>
        </select>
        <select value={league} onChange={e => setLeague(e.target.value)} aria-label="Soccer league">
          <option value="">All leagues</option>
          {SOCCER_LEAGUES.map(x => <option key={x} value={x}>{x}</option>)}
        </select>
        <select value={bucket} onChange={e => setBucket(e.target.value)} aria-label="Entry price bucket">
          <option value="">Any entry price</option>
          {PRICE_BUCKETS.map(x => <option key={x} value={x}>{x}¢</option>)}
        </select>
        <select value={days} onChange={e => setDays(Number(e.target.value))}>
          <option value={14}>14 days</option>
          <option value={30}>30 days</option>
          <option value={60}>60 days</option>
          <option value={90}>90 days</option>
        </select>
      </div>

      {loading && <p className="muted">Loading…</p>}
      {!loading && rows && shown.length === 0 && (
        <p className="muted">No prediction rows match these filters.</p>
      )}
      {!loading && rows && shown.length > 0 && (
        <table className="bets-table" style={{ width: '100%' }}>
          <thead>
            <tr>
              <th>Sector</th>
              <th>Market</th>
              <th>Venue</th>
              <th>Mode</th>
              <th style={{ textAlign: 'right' }} title="clean live-eligible resolved / resolved / logged">n (c/r/l)</th>
              <th style={{ textAlign: 'right' }} title="Paired Brier delta ×1000 — positive = blend beats sharp">ΔBrier/1k</th>
              <th style={{ textAlign: 'right' }} title="Kalshi entry→close CLV: mean pp / % positive (n)">CLV</th>
              <th style={{ textAlign: 'right' }} title="Mean |blended − sharp| in pp. Under 0.5pp on moneyline = sharp passthrough">Div pp</th>
              <th title="clean-n ≥ 30 · CLV n ≥ 30 · mean ≥ 0 · %pos ≥ 55">Gates</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {shown.map(r => {
              const div = r.blend_divergence_pp
              const clv = r.clv
              return (
                <tr
                  key={`${r.sector}-${r.market_type}-${r.venue}`}
                  onClick={() => setOpen(r)}
                  style={{ cursor: 'pointer' }}
                >
                  <td>{r.sector}</td>
                  <td>{MKT_ABBR[r.market_type] ?? r.market_type}</td>
                  <td>{r.venue === 'polymarket_us' ? 'PolyUS' : 'Kalshi'}</td>
                  <td className="muted">{r.mode ?? '?'}</td>
                  <td style={{ textAlign: 'right' }}>
                    {r.n_clean_resolved}/{r.n_resolved}/{r.n_logged}
                    {r.n_not_live_eligible > 0 && (
                      <div
                        className="muted"
                        style={{ fontSize: 11 }}
                        title="Clean resolved rows the scanner would still log as shadow after a promotion (has_full_blend). Not counted by any gate."
                      >
                        −{r.n_not_live_eligible} not live-eligible
                      </div>
                    )}
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    {r.brier_delta_per_1000 == null ? '—' : (
                      <span style={{ color: r.brier_delta_per_1000 >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {r.brier_delta_per_1000 >= 0 ? '+' : ''}{r.brier_delta_per_1000.toFixed(1)}
                      </span>
                    )}
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    {clv.n === 0 ? '—' : (
                      <span title={clv.excluded_stale ? `${clv.excluded_stale} stale-close rows excluded` : undefined}>
                        {clv.mean_clv_pp >= 0 ? '+' : ''}{clv.mean_clv_pp.toFixed(2)} / {(clv.frac_positive * 100).toFixed(0)}% ({clv.n})
                      </span>
                    )}
                  </td>
                  <td style={{ textAlign: 'right', color: r.sharp_passthrough ? 'var(--muted)' : 'var(--green)' }}>
                    {div == null ? '—' : div.toFixed(2)}
                  </td>
                  <td style={{ letterSpacing: 2 }}>
                    {gateOrder.map(k => {
                      const g = r.gates[k]
                      return (
                        <span key={k} title={k} style={{ color: g?.ok ? 'var(--green)' : 'var(--red)' }}>
                          {g?.ok ? '✓' : '✗'}
                        </span>
                      )
                    })}
                  </td>
                  <td>
                    <span style={{ color: verdictColor(r.verdict), fontWeight: 600 }}>
                      {r.verdict}
                    </span>
                    {r.promote_hint && (
                      <div className="muted" style={{ fontSize: 11 }}>
                        {r.promote_hint}{' '}
                        <button
                          type="button"
                          onClick={e => { e.stopPropagation(); copy(r.promote_hint!) }}
                        >Copy</button>
                      </div>
                    )}
                    {r.top_blockers.length > 0 && (
                      <div className="muted" style={{ fontSize: 11 }}>{r.top_blockers.join(' · ')}</div>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginTop: 12, fontSize: 12 }}>
        {(Object.keys(VERDICT_STYLE) as VerdictKind[]).map(k => (
          <span key={k} title={VERDICT_STYLE[k].legend}>
            <span style={{ color: VERDICT_STYLE[k].color, fontWeight: 600 }}>{k}</span>
            <span className="muted"> — {VERDICT_STYLE[k].legend}</span>
          </span>
        ))}
      </div>
      {open && (
        <aside
          role="dialog"
          aria-label="Row details"
          style={{
            position: 'fixed', top: 0, right: 0, bottom: 0, width: 380, maxWidth: '100vw',
            overflowY: 'auto', padding: 16, background: 'var(--bg, #111)',
            borderLeft: '1px solid var(--border, #333)', zIndex: 50,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <h3 style={{ margin: 0, flex: 1 }}>
              {open.sector} · {MKT_ABBR[open.market_type] ?? open.market_type} · {open.venue === 'polymarket_us' ? 'PolyUS' : 'Kalshi'}
            </h3>
            <button type="button" onClick={() => setOpen(null)} aria-label="Close">✕</button>
          </div>
          <p style={{ color: verdictColor(open.verdict), fontWeight: 600 }}>{open.verdict}</p>
          <h4>Gate checklist</h4>
          <ul style={{ paddingLeft: 18 }}>
            {gateOrder.map(k => {
              const g = open.gates[k]
              if (!g) return null
              return (
                <li key={k} style={{ color: g.ok ? 'var(--green)' : 'var(--red)' }}>
                  {g.ok ? '✓' : '✗'} {GATE_LABEL[k] ?? k}: {Number(g.value.toFixed(3))} (need ≥ {g.required})
                </li>
              )
            })}
          </ul>
          <h4>Sample</h4>
          <p className="muted">
            Logged {open.n_logged} · resolved {open.n_resolved} · clean live-eligible {open.n_clean_resolved}
            {open.clv.games != null && <> · CLV games {open.clv.games}</>}
          </p>
          <p>
            <strong>{open.n_not_live_eligible}</strong> clean resolved rows not live-eligible
            <span className="muted"> (would still log shadow after promotion; counted by no gate)</span>
          </p>
          {open.clv.excluded_stale > 0 && (
            <p className="muted">{open.clv.excluded_stale} stale-close rows excluded from CLV.</p>
          )}
          <h4>Top model blockers</h4>
          {open.top_blockers.length === 0
            ? <p className="muted">None recorded (or no diagnostics on these rows).</p>
            : <ul style={{ paddingLeft: 18 }}>{open.top_blockers.map(b => <li key={b}>{b}</li>)}</ul>}
          {open.promote_hint && (
            <>
              <h4>Promote (manual)</h4>
              <pre style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>{open.promote_hint}</pre>
              <button type="button" onClick={() => copy(open.promote_hint!)}>Copy</button>
            </>
          )}
        </aside>
      )}
      <p className="muted" style={{ fontSize: 12, marginTop: 12 }}>
        Div pp = mean |blended − sharp| across logged rows. Moneyline groups under 0.5pp are
        sharp-passthrough: any apparent EV there is venue divergence, not model edge. Gates:
        clean-n ≥ 30 · CLV n ≥ 30 · mean ≥ 0 · %pos ≥ 55 (staleness-filtered on Kalshi).
      </p>
      <ModesPanel />
    </>
  )
}
