import { useCallback, useEffect, useState } from 'react'
import { fetchEval, type EvalParams } from '../lib/api'
import type { EvalBlock, EvalResult } from '../lib/types'

interface Props {
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
}

const POPULATIONS = ['gate', 'clean', 'raw']

// Spec key -> CLI flag, for the copyable command (formatting only).
const FLAGS: [string, string][] = [
  ['sector', '-s'], ['market_type', '-m'], ['venue', '--venue'], ['mode', '--mode'],
  ['league', '--league'], ['side', '--side'], ['sources_token', '--sources-token'],
  ['since', '--since'], ['until', '--until'], ['population', '--population'],
  ['max_staleness_h', '--max-staleness-h'],
]

function cliCommand(spec: Record<string, unknown>): string {
  const parts = ['evmax cleanup eval']
  for (const [k, flag] of FLAGS) {
    const v = spec[k]
    if (v !== null && v !== undefined && v !== '') parts.push(`${flag} ${String(v)}`)
  }
  const gb = spec.group_by
  if (Array.isArray(gb) && gb.length) parts.push(`--group-by ${gb.join(',')}`)
  return parts.join(' ')
}

const fmt = (v: unknown): string => {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') return String(v)
  if (Array.isArray(v) || typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

function KV({ block }: { block: EvalBlock }) {
  return (
    <table className="bets-table" style={{ width: '100%' }}>
      <tbody>
        {Object.entries(block).filter(([k]) => k !== 'uncapped' && k !== 'row_digest' && k !== 'empty_reason').map(([k, v]) => (
          <tr key={k}>
            <td className="muted">{k}</td>
            <td style={{ textAlign: 'right' }}>{fmt(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Digest({ block }: { block: EvalBlock }) {
  return <div className="muted" style={{ fontSize: 11 }}>row_digest {fmt(block.row_digest)}</div>
}

/**
 * Eval tab — renders GET /api/eval (= `evmax cleanup eval`) verbatim. No metric
 * is computed here; the server output is the single source for any quoted number.
 */
export function EvalPage({ toast }: Props) {
  const [p, setP] = useState<EvalParams>({ population: 'gate' })
  const [data, setData] = useState<EvalResult | null>(null)
  const [loading, setLoading] = useState(true)

  const load = useCallback(async (q: EvalParams) => {
    setLoading(true)
    try {
      setData(await fetchEval(q))
    } catch (e) {
      toast('Eval failed: ' + (e as Error).message, 'err')
    } finally {
      setLoading(false)
    }
  }, [toast])

  useEffect(() => { load(p) }, [p, load])

  const set = (k: keyof EvalParams) => (e: { target: { value: string } }) =>
    setP(prev => ({ ...prev, [k]: e.target.value || undefined }))

  const cmd = data ? cliCommand(data.spec) : ''

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
        <h2 style={{ margin: 0 }}>Eval</h2>
        <span className="muted">`evmax cleanup eval`, read-only, unmodified</span>
        <div style={{ flex: 1 }} />
        <input placeholder="sector" size={8} onBlur={set('sector')} defaultValue={p.sector} />
        <input placeholder="market" size={8} onBlur={set('market')} defaultValue={p.market} />
        <select value={p.venue ?? ''} onChange={set('venue')}>
          <option value="">any venue</option>
          <option value="kalshi">kalshi</option>
          <option value="polymarket_us">polymarket_us</option>
        </select>
        <select value={p.mode ?? ''} onChange={set('mode')}>
          <option value="">live + shadow</option>
          <option value="live">live</option>
          <option value="shadow">shadow</option>
        </select>
        <input type="date" value={p.since ?? ''} onChange={set('since')} title="since (event_date ≥)" />
        <select value={p.population ?? 'gate'} onChange={set('population')}>
          {POPULATIONS.map(x => <option key={x} value={x}>{x}</option>)}
        </select>
      </div>

      {loading && <p className="muted">Loading…</p>}
      {data && (
        <>
          <div style={{ marginBottom: 16 }}>
            <div className="muted" style={{ fontSize: 12 }}>Copy into any report (CLAUDE.md Evaluation Policy)</div>
            <code style={{ display: 'block', padding: 8, wordBreak: 'break-all' }}>{cmd}</code>
            <button className="seg" onClick={() => {
              navigator.clipboard.writeText(cmd).then(() => toast('Command copied', 'ok'), () => toast('Copy failed', 'err'))
            }}>Copy command</button>
            <div className="muted" style={{ fontSize: 11, marginTop: 6 }}>db {fmt(data.db)}</div>
          </div>

          {data.groups.map((g, gi) => (
            <section key={gi} style={{ marginBottom: 28 }}>
              {Object.keys(g.group).length > 0 && <h3>{fmt(g.group)}</h3>}

              <h4>Funnel</h4>
              <table className="bets-table" style={{ width: '100%' }}>
                <thead><tr><th>Stage</th><th style={{ textAlign: 'right' }}>Rows</th><th style={{ textAlign: 'right' }}>Dropped</th></tr></thead>
                <tbody>
                  {g.funnel.map(f => (
                    <tr key={f.stage}>
                      <td>{f.stage}</td>
                      <td style={{ textAlign: 'right' }}>{f.n_rows}</td>
                      <td style={{ textAlign: 'right' }}>{f.dropped ?? ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div className="two-col" style={{ marginTop: 16 }}>
                <div>
                  <h4>ROI / Brier population</h4>
                  <div className="muted" style={{ fontSize: 12 }}>
                    Through live_eligible · n_rows {fmt(g.roi_population.n_rows)} · n_games {fmt(g.roi_population.n_games)}
                  </div>
                  {g.roi_empty_reason && <p style={{ color: 'var(--amber)' }}>Empty: {g.roi_empty_reason}</p>}
                  <KV block={g.roi_population} />
                  <Digest block={g.roi_population} />
                </div>
                <div>
                  <h4>CLV population</h4>
                  <div className="muted" style={{ fontSize: 12 }}>
                    Subset with a fresh close · n_rows {fmt(g.clv_population.n_rows)} · n_games {fmt(g.clv_population.n_games)}
                  </div>
                  {g.clv_population.empty_reason && <p style={{ color: 'var(--amber)' }}>Empty: {g.clv_population.empty_reason}</p>}
                  <KV block={g.clv_population} />
                  <Digest block={g.clv_population} />
                  {g.clv_population.uncapped && (
                    <>
                      <h5>Uncapped (no staleness cap)</h5>
                      <KV block={g.clv_population.uncapped} />
                    </>
                  )}
                </div>
              </div>

              <h4 style={{ marginTop: 16 }}>Realized dollars (placed bets, real stake)</h4>
              <div className="muted" style={{ fontSize: 12 }}>
                Quote pnl_est_usd with its bounds pnl_raw_usd … pnl_fee_inclusive_usd and price_kinds.
              </div>
              <KV block={g.realized} />
            </section>
          ))}
        </>
      )}
    </>
  )
}
