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

  const isPlaced = view === 'placed'
  const lineColor = isPlaced ? '#3ddc97' : '#5b9dff'
  const fillColor = isPlaced ? 'rgba(61,220,151,0.12)' : 'rgba(91,157,255,0.14)'

  const chartData = {
    labels: dates,
    datasets: [
      {
        label: isPlaced ? 'Placed Only P&L ($)' : 'All Scanned P&L ($)',
        data: isPlaced ? dataPlaced : dataAll,
        borderColor: lineColor,
        backgroundColor: fillColor,
        fill: true, tension: 0.35, pointRadius: 0, pointHoverRadius: 5, borderWidth: 2.5,
      },
    ],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index' as const, intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#171b25', borderColor: '#232936', borderWidth: 1,
        titleColor: '#aab4c5', bodyColor: '#e7ecf4', padding: 10, cornerRadius: 8,
        displayColors: false,
      },
    },
    scales: {
      x: { ticks: { color: '#6f7c92', maxRotation: 0, autoSkipPadding: 16 }, grid: { display: false }, border: { color: '#232936' } },
      y: { ticks: { color: '#6f7c92', callback: (v: unknown) => '$' + v }, grid: { color: 'rgba(255,255,255,0.04)' }, border: { display: false } },
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
