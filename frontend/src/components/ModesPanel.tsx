import { useEffect, useState } from 'react'
import { fetchModes } from '../lib/api'
import type { ModesReport } from '../lib/types'

const REASON_LABEL: Record<string, string> = {
  yaml_base: 'YAML base',
  runtime_override: 'runtime override',
  env_override: 'env override',
  off_season_downgrade: 'off-season downgrade',
  market_type_shadow: 'market-type shadow',
  market_type_disabled: 'market-type disabled',
  venue_market_type_shadow: 'venue market-type shadow',
  venue_firewall: 'venue firewall',
}

const MODE_COLOR: Record<string, string> = {
  live: 'var(--green)',
  shadow: 'var(--amber)',
  disabled: 'var(--muted)',
}

const MKT_ABBR: Record<string, string> = {
  moneyline: 'ML', spread: 'Spread', total: 'Total', advance: 'Advance', player_prop: 'Prop',
}

/** Read-only effective-mode view (`evmax categories modes`, plus WHY). */
export function ModesPanel() {
  const [data, setData] = useState<ModesReport | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [onlyOverrides, setOnlyOverrides] = useState(false)

  useEffect(() => {
    fetchModes().then(setData).catch(e => setErr((e as Error).message))
  }, [])

  if (err) return <p className="muted">Modes load failed: {err}</p>
  if (!data) return null
  const rows = onlyOverrides ? data.rows.filter(r => r.reason !== 'yaml_base') : data.rows
  const pu = data.polymarket_us

  return (
    <section style={{ marginTop: 32 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', marginBottom: 8 }}>
        <h2 style={{ margin: 0 }}>Modes</h2>
        <span className="muted">Effective mode per category · market · venue, and why (read-only)</span>
        <div style={{ flex: 1 }} />
        <label className="muted" style={{ fontSize: 12 }}>
          <input type="checkbox" checked={onlyOverrides} onChange={e => setOnlyOverrides(e.target.checked)} />{' '}
          only non-default
        </label>
      </div>
      <p className="muted" style={{ fontSize: 12, margin: '0 0 8px' }}>
        PolyUS firewall: {pu.enabled ? (pu.master_live ? 'master switch ON (all sectors cleared)' : `allowlist ${pu.live_sectors.join(', ') || '(none)'}`) : 'venue disabled'}
        {' · '}Shadow soccer leagues: {data.shadow_leagues.join(', ') || 'none'}
      </p>
      <table className="bets-table" style={{ width: '100%' }}>
        <thead>
          <tr>
            <th>Category</th><th>Market</th><th>Venue</th><th>YAML</th><th>Effective</th><th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(r => (
            <tr key={`${r.category}-${r.market_type}-${r.venue}`}>
              <td>{r.display_name}</td>
              <td>{MKT_ABBR[r.market_type] ?? r.market_type}</td>
              <td>{r.venue === 'polymarket_us' ? 'PolyUS' : 'Kalshi'}</td>
              <td className="muted">{r.yaml_mode}</td>
              <td style={{ color: MODE_COLOR[r.mode] ?? 'inherit', fontWeight: 600 }}>{r.mode}</td>
              <td className="muted">{REASON_LABEL[r.reason] ?? r.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
