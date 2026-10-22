import { useState, useEffect, useCallback, useRef } from 'react'
import type { Summary, ProfitPoint, SectorRow, Bet, ScanGap } from '../lib/types'
import { fetchDashboard } from '../lib/api'

export interface DashboardState {
  summaryAll: Summary | null
  summaryPlaced: Summary | null
  seriesAll: ProfitPoint[]
  seriesPlaced: ProfitPoint[]
  sectors: SectorRow[]
  placedBets: Bet[]
  openBets: Bet[]
  recent: Bet[]
  scanGaps: ScanGap[]
  scanMeta: { markets_fetched: number; markets_matched: number } | null
  scanBankroll: number
  scanKelly: number
  loading: boolean
}

export const POLL_INTERVAL_MS = 30_000

/** `active` = dashboard page is showing; polling only runs while it is true. */
export function useDashboard(active = true) {
  const [state, setState] = useState<DashboardState>({
    summaryAll: null,
    summaryPlaced: null,
    seriesAll: [],
    seriesPlaced: [],
    sectors: [],
    placedBets: [],
    openBets: [],
    recent: [],
    scanGaps: [],
    scanMeta: null,
    scanBankroll: 250,
    scanKelly: 0.5,
    loading: true,
  })

  const inFlight = useRef(false)
  const [updatedAt, setUpdatedAt] = useState<number | null>(null)
  const [paused, setPaused] = useState(false)

  const refresh = useCallback(() => {
    if (inFlight.current) return Promise.resolve()
    inFlight.current = true
    return fetchDashboard()
      .then(data => {
        setUpdatedAt(Date.now())
        setState(prev => {
          const excluded = new Set<string>([
            ...data.placed_bets.map(b => b.market_id),
            ...data.recent.map(b => b.market_id),
          ])
          return {
            ...prev,
            summaryAll: data.summary_all,
            summaryPlaced: data.summary_placed,
            seriesAll: data.series_all,
            seriesPlaced: data.series_placed,
            sectors: data.sectors,
            placedBets: data.placed_bets,
            openBets: data.unplaced_bets,
            recent: data.recent,
            scanGaps: prev.scanGaps.filter(g => !excluded.has(g.market_id)),
            loading: false,
          }
        })
      })
      .catch((e: unknown) => {
        console.error('Failed to load dashboard', e)
        setState(prev => ({ ...prev, loading: false }))
      })
      .finally(() => { inFlight.current = false })
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  // Visibility-aware poll: no timer while the tab is hidden, paused by the
  // user, or off the dashboard page; refreshes immediately on becoming visible.
  useEffect(() => {
    if (!active || paused) return
    let timer: ReturnType<typeof setInterval> | null = null
    const stop = () => { if (timer !== null) { clearInterval(timer); timer = null } }
    const start = () => {
      stop()
      timer = setInterval(() => { void refresh() }, POLL_INTERVAL_MS)
    }
    const onVisibility = () => {
      if (document.hidden) { stop() } else { void refresh(); start() }
    }
    if (!document.hidden) start()
    document.addEventListener('visibilitychange', onVisibility)
    window.addEventListener('focus', onVisibility)
    return () => {
      stop()
      document.removeEventListener('visibilitychange', onVisibility)
      window.removeEventListener('focus', onVisibility)
    }
  }, [active, paused, refresh])

  const setScanResults = useCallback((
    gaps: ScanGap[],
    meta: { markets_fetched: number; markets_matched: number },
    snapshot: { bankroll: number; kelly: number },
  ) => {
    setState(prev => {
      const excluded = new Set<string>([
        ...prev.placedBets.map(b => b.market_id),
        ...prev.recent.map(b => b.market_id),
      ])
      return {
        ...prev,
        scanGaps: gaps.filter(g => !excluded.has(g.market_id)),
        scanMeta: meta,
        scanBankroll: snapshot.bankroll,
        scanKelly: snapshot.kelly,
      }
    })
  }, [])

  return { ...state, refresh, setScanResults, updatedAt, paused, setPaused }
}
