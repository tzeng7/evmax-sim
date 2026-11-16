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
  // Epoch ms of the scan that produced scanGaps (null = no scan this session).
  scanAt: number | null
  loading: boolean
}

export const POLL_INTERVAL_MS = 30_000

/** `active` = dashboard page is showing; polling only runs while it is true. */
// The last scan survives a reload (sessionStorage, this tab only). The panel
// shows its age, so a restored scan is never mistaken for a fresh one.
const SCAN_KEY = 'evmax.lastScan'
const SCAN_MAX_AGE_MS = 6 * 60 * 60 * 1000

interface StoredScan {
  gaps: ScanGap[]
  meta: { markets_fetched: number; markets_matched: number }
  bankroll: number
  kelly: number
  at: number
}

function loadStoredScan(): StoredScan | null {
  try {
    const raw = window.sessionStorage.getItem(SCAN_KEY)
    if (!raw) return null
    const s = JSON.parse(raw) as StoredScan
    if (!Array.isArray(s.gaps) || typeof s.at !== 'number') return null
    if (Date.now() - s.at > SCAN_MAX_AGE_MS) return null
    return s
  } catch {
    return null
  }
}

export function useDashboard(active = true) {
  const [state, setState] = useState<DashboardState>(() => {
    const stored = loadStoredScan()
    return {
    summaryAll: null,
    summaryPlaced: null,
    seriesAll: [],
    seriesPlaced: [],
    sectors: [],
    placedBets: [],
    openBets: [],
    recent: [],
    scanGaps: stored?.gaps ?? [],
    scanMeta: stored?.meta ?? null,
    scanBankroll: stored?.bankroll ?? 250,
    scanKelly: stored?.kelly ?? 0.5,
    scanAt: stored?.at ?? null,
    loading: true,
    }
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
    const at = Date.now()
    try {
      const stored: StoredScan = { gaps, meta, bankroll: snapshot.bankroll, kelly: snapshot.kelly, at }
      window.sessionStorage.setItem(SCAN_KEY, JSON.stringify(stored))
    } catch { /* storage full or unavailable: the scan still shows, it just won't survive a reload */ }
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
        scanAt: at,
      }
    })
  }, [])

  return { ...state, refresh, setScanResults, updatedAt, paused, setPaused }
}
