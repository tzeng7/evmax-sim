import { useEffect, useMemo, useState } from 'react'
import type { ProjectionSizing } from './types'
import { validOdds, type StakeContext } from './projections'

const SAVED = 'evmax.projections.sizing'

export interface SizingSettings {
  bankroll: number | null
  preset: string
  customOdds: number
  kelly: number | null
}

function load(): SizingSettings {
  try {
    const raw = localStorage.getItem(SAVED)
    if (raw) return { bankroll: null, preset: 'book', customOdds: -110, kelly: null, ...JSON.parse(raw) }
  } catch { /* storage blocked or bad JSON: start blank */ }
  return { bankroll: null, preset: 'book', customOdds: -110, kelly: null }
}

/** Bankroll / price / Kelly settings, remembered per browser; null context until a bankroll is entered. */
export function useStakeContext(sizing: ProjectionSizing | null | undefined) {
  const [settings, setSettings] = useState<SizingSettings>(load)
  useEffect(() => {
    try { localStorage.setItem(SAVED, JSON.stringify(settings)) } catch { /* storage blocked: settings last this visit */ }
  }, [settings])

  const ctx = useMemo<StakeContext | null>(() => {
    if (!sizing || !settings.bankroll || settings.bankroll <= 0) return null
    const preset = sizing.presets.find(p => p.key === settings.preset)
    const odds = preset ? preset.odds : settings.customOdds
    if (!validOdds(odds)) return null
    return {
      bankroll: settings.bankroll, odds, oddsLabel: preset ? preset.label : `${odds > 0 ? '+' : ''}${odds}`,
      kelly: settings.kelly ?? sizing.default_kelly, cap: sizing.cap,
    }
  }, [sizing, settings])
  return { settings, setSettings, ctx }
}

