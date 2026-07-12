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
  scanBankroll: number
  scanKelly: number
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
    scanBankroll: 250,
    scanKelly: 0.5,
    loading: true,
  })

  const refresh = useCallback(() => {
    return fetchDashboard()
      .then(data => {
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
  }, [])

  useEffect(() => { void refresh() }, [refresh])

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

  return { ...state, refresh, setScanResults }
}
