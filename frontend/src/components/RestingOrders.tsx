import { useCallback, useEffect, useState } from 'react'
import { VenueLogo } from './VenueLogo'
import { recordMakerFills } from '../lib/api'
import { probToCents } from '../lib/odds'

// RECORD-ONLY: this panel never places or cancels exchange orders. It lists the
// maker rows the user should have resting, counts down to kickoff, warns when a
// game has started with an order still unfilled, and records fills via /api/fill.

interface RestingOrder {
  market_id: string
  sector: string
  event_title: string
  yes_team: string
  market_type: string
  line: number | null
  venue: string
  pilot: boolean
  status: 'resting' | 'filled' | 'past_kickoff'
  kickoff_at: string | null
  age_minutes: number | null
  ask_price: number | null
  maker_ev_pct: number | null
  placed_price: number | null
  placed_stake: number | null
}

interface PilotReadout {
  sector: string
  enabled: boolean
  kelly_mult: number
  checkpoint_games: number
  gate_games: number
  candidates_resolved: number
  candidates_filled: number
  candidates_open: number
  fill_rate: number | null
  games: number
  rows: number
  mean_net_pp: number | null
  z: number | null
  frac_positive: number | null
  verdict: string
  action: string
}

interface RestingResponse { orders: RestingOrder[]; readout: PilotReadout }

interface Props {
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
  onChanged?: () => void
}

function fmtDuration(totalSec: number): string {
  const s = Math.abs(Math.round(totalSec))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  return h > 0 ? `${h}h ${String(m).padStart(2, '0')}m` : `${m}m ${String(sec).padStart(2, '0')}s`
}

function fmtAge(min: number | null): string {
  if (min == null) return '—'
  return min >= 60 ? `${Math.floor(min / 60)}h ${min % 60}m` : `${min}m`
}

function outcome(o: RestingOrder): string {
  const line = o.line == null ? '' : ` ${o.line > 0 ? '+' : ''}${o.line}`
  return `${o.yes_team}${line}`
}

export function RestingOrders({ toast, onChanged }: Props) {
  const [data, setData] = useState<RestingResponse | null>(null)
  const [now, setNow] = useState(() => Date.now())
  const [inputs, setInputs] = useState<Record<string, { price: string; stake: string }>>({})
  const [busy, setBusy] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const res = await fetch('/api/resting-orders')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setData(await res.json())
    } catch (e) {
      toast('Resting orders failed: ' + (e as Error).message, 'err')
    }
  }, [toast])

  useEffect(() => { load() }, [load])
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [])
  // Re-fetch every minute so a row flips to past_kickoff / filled without a reload.
  useEffect(() => {
    const t = setInterval(load, 60_000)
    return () => clearInterval(t)
  }, [load])

  const setInput = (mid: string, field: 'price' | 'stake', value: string) =>
    setInputs(prev => ({ ...prev, [mid]: { ...(prev[mid] ?? { price: '', stake: '' }), [field]: value } }))

  const recordFill = async (o: RestingOrder) => {
    const inp = inputs[o.market_id]
    const price = Number(inp?.price || (o.ask_price != null ? o.ask_price * 100 : NaN))
    const stake = Number(inp?.stake)
    if (!Number.isFinite(price) || !(stake > 0)) {
      toast('Enter the fill price (¢) and stake ($) first', 'err')
      return
    }
    setBusy(o.market_id)
    try {
      const res = await recordMakerFills([{ market_id: o.market_id, fill_price: price, fill_stake: stake }])
      if (res.filled > 0) {
        toast('Fill recorded', 'ok')
        await load()
        onChanged?.()
      } else {
        toast(`Not recorded — ${res.skipped?.[0]?.reason ?? 'unknown reason'}`, 'err')
      }
    } catch (e) {
      toast('Fill failed: ' + (e as Error).message, 'err')
    } finally {
      setBusy(null)
    }
  }

  if (!data) return null
  const { orders, readout: r } = data
  if (orders.length === 0 && !r.enabled && r.games === 0) return null

  const warnCount = orders.filter(o => o.status === 'past_kickoff').length
  const pct = (x: number | null) => (x == null ? '—' : `${(x * 100).toFixed(0)}%`)

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Resting orders ({orders.filter(o => o.status !== 'filled').length})</h2>
        <span className="muted" style={{ fontSize: 12 }}>Record-only — place and cancel orders on the exchange</span>
      </div>

      <div style={{ padding: '8px 12px', display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'baseline' }}>
        <strong>Maker pilot · {r.sector.toUpperCase()}</strong>
        <span className={`badge ${r.verdict === 'STOP' ? 'warn' : ''}`} title={r.action}>{r.verdict}</span>
        <span>Filled games <strong>{r.games}</strong> / {r.checkpoint_games} checkpoint · {r.gate_games} gate</span>
        <span>Net CLV/game (after maker fee){' '}
          <strong className={r.mean_net_pp != null && r.mean_net_pp < 0 ? 'red' : 'green'}>
            {r.mean_net_pp == null ? '—' : `${r.mean_net_pp >= 0 ? '+' : ''}${r.mean_net_pp.toFixed(2)}pp`}
          </strong>
          {r.z != null && <span className="muted"> (z {r.z.toFixed(2)})</span>}
        </span>
        <span>Rows positive <strong>{pct(r.frac_positive)}</strong> (need 55%)</span>
        <span>Fill rate <strong>{pct(r.fill_rate)}</strong>
          <span className="muted"> ({r.candidates_filled}/{r.candidates_resolved} resolved)</span></span>
        <span className="muted">{r.action}</span>
      </div>

      {warnCount > 0 && (
        <div role="alert" style={{ padding: '8px 12px', color: 'var(--red)', fontWeight: 600 }}>
          {warnCount} order(s) are past kickoff and unfilled — cancel on exchange now (Kalshi trades in-play).
        </div>
      )}

      {orders.length > 0 && (
        <div className="scroll-area">
          <table>
            <thead>
              <tr>
                <th>Status</th><th>Sector</th><th style={{ width: 34 }}>Venue</th><th>Event</th><th>Outcome</th>
                <th className="num">Ask</th><th className="num">Maker EV</th>
                <th className="num">Kickoff</th><th className="num">Age</th><th>Record fill</th>
              </tr>
            </thead>
            <tbody>
              {orders.map(o => {
                const kick = o.kickoff_at ? new Date(o.kickoff_at).getTime() : null
                const remaining = kick != null ? (kick - now) / 1000 : null
                const inp = inputs[o.market_id]
                return (
                  <tr key={o.market_id}>
                    <td>
                      {o.status === 'resting' && <span className="badge">RESTING</span>}
                      {o.status === 'filled' && <span className="badge new">FILLED</span>}
                      {o.status === 'past_kickoff' && (
                        <span className="badge warn" title="Game has started; cancel the unfilled order on the exchange">
                          CANCEL ON EXCHANGE
                        </span>
                      )}
                      {o.pilot && <span className="badge" title="Maker pilot row"> PILOT</span>}
                    </td>
                    <td><span className="badge">{o.sector}</span></td>
                    <td><VenueLogo venue={o.venue} /></td>
                    <td>{o.event_title}</td>
                    <td>{outcome(o)}</td>
                    <td className="num">{probToCents(o.ask_price)}</td>
                    <td className="num green">{o.maker_ev_pct == null ? '—' : `${(o.maker_ev_pct * 100).toFixed(1)}%`}</td>
                    <td className="num">
                      {remaining == null ? '—'
                        : remaining >= 0 ? `in ${fmtDuration(remaining)}`
                        : <span className="red">{fmtDuration(remaining)} ago</span>}
                    </td>
                    <td className="num">{fmtAge(o.age_minutes)}</td>
                    <td>
                      {o.status === 'filled' ? (
                        <span className="muted">
                          {o.placed_price != null ? `${probToCents(o.placed_price)}` : ''}
                          {o.placed_stake != null ? ` · $${o.placed_stake.toFixed(2)}` : ''}
                        </span>
                      ) : (
                        <span style={{ display: 'inline-flex', gap: 4 }}>
                          <input
                            type="number" aria-label="Fill price in cents" placeholder="¢"
                            style={{ width: 56 }}
                            value={inp?.price ?? ''}
                            onChange={e => setInput(o.market_id, 'price', e.target.value)}
                          />
                          <input
                            type="number" aria-label="Stake in dollars" placeholder="$"
                            style={{ width: 64 }}
                            value={inp?.stake ?? ''}
                            onChange={e => setInput(o.market_id, 'stake', e.target.value)}
                          />
                          <button className="btn btn-sm success" disabled={busy === o.market_id}
                            onClick={() => recordFill(o)}>Record fill</button>
                        </span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
