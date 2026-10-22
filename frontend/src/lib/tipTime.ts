import { useEffect, useState } from 'react'

// Entry-timing buckets. Repo notes: only the under-1h-pre-tip entry window
// carries positive CLV, so a night-before edge is a watchlist item, not a pick.
export type TipBucket = 'watch' | 'pick' | 'late' | 'unknown'

export const WATCH_MIN = 120   // > 2h to tip
export const PICK_FAR_MIN = 90 // pick window opens at T-90
export const PICK_NEAR_MIN = 15 // ...and closes at T-15

export const BUCKET_ORDER: TipBucket[] = ['pick', 'watch', 'unknown', 'late']

export const BUCKET_LABEL: Record<TipBucket, string> = {
  watch: 'Watchlist (>2h)',
  pick: 'Pick window (T-90 to T-15)',
  late: 'Late / in-play',
  unknown: 'Tip time unknown',
}

/** Minutes until tip (negative once started); null when the tip time is unknown. */
export function minutesToTip(tip: string | null | undefined, nowMs: number): number | null {
  if (!tip) return null
  const t = Date.parse(tip)
  return Number.isNaN(t) ? null : (t - nowMs) / 60000
}

// T-90..T-120 is a deliberate gap: too late for the watchlist, too early to pick.
// It stays in 'watch' so nothing disappears between buckets.
export function tipBucket(mins: number | null): TipBucket {
  if (mins === null) return 'unknown'
  if (mins < PICK_NEAR_MIN) return 'late'
  if (mins <= PICK_FAR_MIN) return 'pick'
  return 'watch'
}

export function formatTimeToTip(mins: number | null): string {
  if (mins === null) return '—'
  if (mins < 0) return `started ${formatSpan(-mins)} ago`
  return formatSpan(mins)
}

function formatSpan(mins: number): string {
  const m = Math.floor(mins)
  if (m < 60) return `${m}m`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ${m % 60}m`
  return `${Math.floor(h / 24)}d ${h % 24}h`
}

/** Wall-clock ms that re-renders the caller every `everyMs` (default 30s). */
export function useNow(everyMs = 30000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), everyMs)
    return () => clearInterval(id)
  }, [everyMs])
  return now
}

export function bucketStyle(b: TipBucket): React.CSSProperties {
  if (b === 'pick') return { color: '#98c379', fontWeight: 600 }
  if (b === 'late') return { color: '#e5c07b' }
  return {}
}
