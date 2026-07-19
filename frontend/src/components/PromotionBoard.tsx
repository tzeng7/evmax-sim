import { useEffect, useState } from 'react'
import { fetchPromotionBoard } from '../lib/api'
import type { BoardRow } from '../lib/types'

interface Props {
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
}

const VERDICT_COLOR: Record<string, string> = {
  'SHARP-PASSTHROUGH': 'var(--red)',
  'LIVE-DEGRADING': 'var(--red)',
  'FAILING-CLV': 'var(--amber)',
  'PROMOTE-READY': 'var(--green)',
  'LIVE-HEALTHY': 'var(--green)',
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
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetchPromotionBoard(days)
      .then(res => { if (!cancelled) setRows(res.rows) })
      .catch(e => toast('Board load failed: ' + (e as Error).message, 'err'))
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [days, toast])

  const gateOrder = ['clean_n', 'clv_n', 'clv_mean', 'clv_frac_pos']

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
        <h2 style={{ margin: 0 }}>Promotion Board</h2>
        <span className="muted">
          Blend divergence vs sharp + CLV promotion gates, per sector · market · venue
        </span>
        <div style={{ flex: 1 }} />
        <select value={days} onChange={e => setDays(Number(e.target.value))}>
          <option value={14}>14 days</option>
          <option value={30}>30 days</option>
          <option value={60}>60 days</option>
          <option value={90}>90 days</option>
        </select>
      </div>

      {loading && <p className="muted">Loading…</p>}
      {!loading && rows && rows.length === 0 && (
        <p className="muted">No prediction rows in the window.</p>
      )}
      {!loading && rows && rows.length > 0 && (
        <table className="bets-table" style={{ width: '100%' }}>
          <thead>
            <tr>
              <th>Sector</th>
              <th>Market</th>
              <th>Venue</th>
              <th>Mode</th>
              <th style={{ textAlign: 'right' }} title="clean resolved / resolved / logged">n (c/r/l)</th>
              <th style={{ textAlign: 'right' }} title="Paired Brier delta ×1000 — positive = blend beats sharp">ΔBrier/1k</th>
              <th style={{ textAlign: 'right' }} title="Kalshi entry→close CLV: mean pp / % positive (n)">CLV</th>
              <th style={{ textAlign: 'right' }} title="Mean |blended − sharp| in pp. Under 0.5pp on moneyline = sharp passthrough">Div pp</th>
              <th title="clean-n ≥ 30 · CLV n ≥ 30 · mean ≥ 0 · %pos ≥ 55">Gates</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => {
              const div = r.blend_divergence_pp
              const clv = r.clv
              return (
                <tr key={`${r.sector}-${r.market_type}-${r.venue}`}>
                  <td>{r.sector}</td>
                  <td>{MKT_ABBR[r.market_type] ?? r.market_type}</td>
                  <td>{r.venue === 'polymarket_us' ? 'PolyUS' : 'Kalshi'}</td>
                  <td className="muted">{r.mode ?? '?'}</td>
                  <td style={{ textAlign: 'right' }}>
                    {r.n_clean_resolved}/{r.n_resolved}/{r.n_logged}
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
                    <span style={{ color: VERDICT_COLOR[r.verdict] ?? 'var(--muted)', fontWeight: 600 }}>
                      {r.verdict}
                    </span>
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
      <p className="muted" style={{ fontSize: 12, marginTop: 12 }}>
        Div pp = mean |blended − sharp| across logged rows. Moneyline groups under 0.5pp are
        sharp-passthrough: any apparent EV there is venue divergence, not model edge. Gates:
        clean-n ≥ 30 · CLV n ≥ 30 · mean ≥ 0 · %pos ≥ 55 (staleness-filtered on Kalshi).
      </p>
    </>
  )
}
