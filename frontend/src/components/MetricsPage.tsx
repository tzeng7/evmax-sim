import { useEffect, useState, useCallback } from 'react'
import { fetchMetrics } from '../lib/api'
import type { MetricsResult } from '../lib/types'
import { MetricsPanel } from './MetricsPanel'

// Lookback windows for the metrics view. The /api/metrics endpoint takes a
// week count; "1y" is just a wide window so all-but-ancient rows are included.
const RANGES: { label: string; weeks: number }[] = [
  { label: '4w', weeks: 4 },
  { label: '12w', weeks: 12 },
  { label: '26w', weeks: 26 },
  { label: '1y', weeks: 52 },
]

interface Props {
  toast: (msg: string, type?: 'info' | 'ok' | 'err') => void
}

export function MetricsPage({ toast }: Props) {
  const [weeks, setWeeks] = useState(4)
  const [loading, setLoading] = useState(true)
  const [metrics, setMetrics] = useState<MetricsResult | null>(null)

  const load = useCallback(async (w: number) => {
    setLoading(true)
    try {
      setMetrics(await fetchMetrics(w))
    } catch (e) {
      toast('Metrics failed: ' + (e as Error).message, 'err')
    } finally {
      setLoading(false)
    }
  }, [toast])

  useEffect(() => { load(weeks) }, [weeks, load])

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>Model Metrics</h2>
        <span className="muted">Brier, CLV and EV% across all sectors</span>
        <div style={{ flex: 1 }} />
        <div className="segmented">
          {RANGES.map(r => (
            <button
              key={r.label}
              className={`seg ${weeks === r.weeks ? 'active' : ''}`}
              onClick={() => setWeeks(r.weeks)}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      {loading && <div className="skeleton" style={{ height: 320 }} />}
      {!loading && <MetricsPanel data={metrics} />}
    </>
  )
}
