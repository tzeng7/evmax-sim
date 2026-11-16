import { useMemo, useState } from 'react'
import type { SectorRow } from '../lib/types'
import { fmtInt, fmtPnl, fmtSignedPct } from '../lib/format'

interface Props { sectors: SectorRow[] }

type Key = 'sector' | 'bets' | 'win_rate' | 'pnl' | 'roi_pct'

export function SectorPerformance({ sectors }: Props) {
  const [sort, setSort] = useState<{ key: Key; dir: 1 | -1 }>({ key: 'pnl', dir: -1 })

  const rows = useMemo(() => [...sectors].sort((a, b) => {
    const x = a[sort.key], y = b[sort.key]
    return (x < y ? -1 : x > y ? 1 : 0) * sort.dir
  }), [sectors, sort])

  const th = (key: Key, label: string, numeric = true) => {
    const active = sort.key === key
    return (
      <th className={numeric ? 'num' : ''} aria-sort={active ? (sort.dir === 1 ? 'ascending' : 'descending') : undefined}>
        <button type="button" className={`th-sort ${active ? 'active' : ''}`}
          onClick={() => setSort(prev => prev.key === key ? { key, dir: prev.dir === 1 ? -1 : 1 } : { key, dir: numeric ? -1 : 1 })}>
          {label}<span className="th-arrow" aria-hidden="true">{active ? (sort.dir === 1 ? '↑' : '↓') : ''}</span>
        </button>
      </th>
    )
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <h2>By sector</h2>
          <span className="panel-sub">all resolved rows, live and shadow</span>
        </div>
      </div>
      <div className="scroll-area table-scroll">
        <table>
          <thead>
            <tr>{th('sector', 'Sector', false)}{th('bets', 'Bets')}{th('win_rate', 'Win %')}{th('pnl', 'P&L')}{th('roi_pct', 'ROI')}</tr>
          </thead>
          <tbody>
            {rows.map(s => (
              <tr key={s.sector}>
                <td><span className="badge">{s.sector.replace(/_/g, ' ')}</span></td>
                <td className="num" title={`${fmtInt(s.wins)} wins`}>{fmtInt(s.bets)}</td>
                <td className="num">{Number(s.win_rate).toFixed(1)}%</td>
                <td className={`num ${s.pnl >= 0 ? 'green' : 'red'}`}>{fmtPnl(s.pnl)}</td>
                <td className={`num ${s.roi_pct >= 0 ? 'green' : 'red'}`}>{fmtSignedPct(Number(s.roi_pct))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
