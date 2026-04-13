import { useMemo, useState } from 'react'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Title, Tooltip, Legend, Filler,
} from 'chart.js'
import type { ProfitPoint } from '../lib/types'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler)

const RANGES = [
  { label: '1D', days: 1 },
  { label: '1W', days: 7 },
  { label: '1M', days: 30 },
  { label: '3M', days: 90 },
  { label: '1Y', days: 365 },
  { label: 'All', days: 0 },
]

interface Props {
  seriesAll: ProfitPoint[]
  seriesPlaced: ProfitPoint[]
  view: 'all' | 'placed'
}

export function ProfitChart({ seriesAll, seriesPlaced, view }: Props) {
  const [range, setRange] = useState(0)

  const { dates, dataAll, dataPlaced } = useMemo(() => {
    // Merge dates
    const allDates = [...new Set([...seriesAll.map(p => p.date), ...seriesPlaced.map(p => p.date)])].sort()
    const allMap = Object.fromEntries(seriesAll.map(p => [p.date, p.cumulative]))
    const placedMap = Object.fromEntries(seriesPlaced.map(p => [p.date, p.cumulative]))

    let lastAll = 0, lastPlaced = 0
    const filledAll: number[] = []
    const filledPlaced: number[] = []
    for (const d of allDates) {
      if (d in allMap) lastAll = allMap[d]
      if (d in placedMap) lastPlaced = placedMap[d]
      filledAll.push(lastAll)
      filledPlaced.push(lastPlaced)
    }

    // Apply range filter
    if (range > 0) {
      const cutoffDays = range === 1 ? 0 : range
      const cutoff = new Date(Date.now() - cutoffDays * 86400000).toISOString().slice(0, 10)
      const startIdx = allDates.findIndex(d => d >= cutoff)
      if (startIdx > 0) {
        const dates = allDates.slice(startIdx)
        const baseAll = filledAll[startIdx - 1] || 0
        const basePlaced = filledPlaced[startIdx - 1] || 0
        return {
          dates,
          dataAll: filledAll.slice(startIdx).map(v => +(v - baseAll).toFixed(2)),
          dataPlaced: filledPlaced.slice(startIdx).map(v => +(v - basePlaced).toFixed(2)),
        }
      }
    }
    return { dates: allDates, dataAll: filledAll, dataPlaced: filledPlaced }
  }, [seriesAll, seriesPlaced, range])

  const chartData = {
    labels: dates,
    datasets: [
      {
        label: 'All Scanned P&L ($)',
        data: dataAll,
        borderColor: '#60a5fa',
        backgroundColor: 'rgba(96,165,250,0.1)',
        fill: true, tension: 0.3, pointRadius: 2, pointHoverRadius: 5, borderWidth: 2,
        hidden: view === 'placed',
      },
      {
        label: 'Placed Only P&L ($)',
        data: dataPlaced,
        borderColor: '#4ade80',
        backgroundColor: 'rgba(74,222,128,0.08)',
        fill: true, tension: 0.3, pointRadius: 2, pointHoverRadius: 5, borderWidth: 2,
        hidden: view === 'all',
      },
    ],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#d6deeb' } },
      tooltip: { mode: 'index' as const, intersect: false },
    },
    scales: {
      x: { ticks: { color: '#7a8aa0' }, grid: { color: '#1f2530' } },
      y: { ticks: { color: '#7a8aa0', callback: (v: unknown) => '$' + v }, grid: { color: '#1f2530' } },
    },
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Cumulative Profit</h2>
        <div className="actions" style={{ gap: 4 }}>
          {RANGES.map(r => (
            <button
              key={r.days}
              className={`btn btn-sm range-btn ${range === r.days ? 'active' : ''}`}
              onClick={() => setRange(r.days)}
            >{r.label}</button>
          ))}
        </div>
      </div>
      <div style={{ height: 280 }}>
        <Line data={chartData} options={options} />
      </div>
    </div>
  )
}
