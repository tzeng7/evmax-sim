import { useState, useEffect, useCallback } from 'react'
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
  loading: boolean
}

export function useDashboard() {
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
    loading: true,
  })

  const refresh = useCallback(async () => {
    try {
      const data = await fetchDashboard()
      setState(prev => ({
        ...prev,
        summaryAll: data.summary_all,
        summaryPlaced: data.summary_placed,
        seriesAll: data.series_all,
        seriesPlaced: data.series_placed,
        sectors: data.sectors,
        placedBets: data.placed_bets,
        openBets: data.unplaced_bets,
        recent: data.recent,
        loading: false,
      }))
    } catch (e) {
      console.error('Failed to load dashboard', e)
      setState(prev => ({ ...prev, loading: false }))
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const setScanResults = useCallback((gaps: ScanGap[], meta: { markets_fetched: number; markets_matched: number }) => {
    setState(prev => ({ ...prev, scanGaps: gaps, scanMeta: meta }))
  }, [])

  return { ...state, refresh, setScanResults }
}
