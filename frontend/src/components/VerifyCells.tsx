import type { VerifyRow } from '../lib/api'
import { probToCents } from '../lib/odds'

// Shared "Live ask / Δ" cells for the Verify-live flow (ScanResults + OpenPositions).
export function VerifyHeaderCells() {
  return (
    <>
      <th className="num" title="Live ask from the last Verify live (scan ask until verified)">Live</th>
      <th className="num" title="Live ask minus scan ask, in cents. Positive = the edge is eroding.">Δ</th>
    </>
  )
}

export function VerifyCells({ v }: { v?: VerifyRow }) {
  if (!v) return <><td className="num muted">—</td><td className="num muted">—</td></>
  if (v.live_ask == null) {
    return <td className="num red" colSpan={2} title={v.reason}>{v.in_play ? 'in-play' : 'no quote'}</td>
  }
  const d = (v.delta ?? 0) * 100
  const color = d > 0.5 ? 'red' : d < -0.5 ? 'green' : 'muted'
  return (
    <>
      <td className="num" title={v.reason}>{probToCents(v.live_ask)}</td>
      <td className={`num ${color}`} title={v.live_ev != null ? `Live EV ${(v.live_ev * 100).toFixed(1)}%` : v.reason}>
        {d > 0 ? '+' : ''}{d.toFixed(0)}¢
      </td>
    </>
  )
}
