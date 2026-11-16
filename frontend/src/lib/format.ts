// Shared display formatters. One place for money / dates / labels so every
// table on the dashboard reads the same way.

const usd = new Intl.NumberFormat('en-US', {
  style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2,
})
const int = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 })

/** "$1,157.98" — unsigned amounts (stakes, balances, exposure). */
export function fmtUsd(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—'
  return usd.format(n)
}

/** "+$6.42" / "−$18.77" — signed P&L. The sign leads the currency symbol. */
export function fmtPnl(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—'
  const abs = usd.format(Math.abs(n))
  if (Math.abs(n) < 0.005) return abs
  return (n > 0 ? '+' : '−') + abs
}

/** "28,626" */
export function fmtInt(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—'
  return int.format(n)
}

/** "+6.7%" / "−16.5%" — signed percentage already expressed in percent units. */
export function fmtSignedPct(n: number | null | undefined, digits = 1): string {
  if (n == null || !Number.isFinite(n)) return '—'
  const abs = Math.abs(n).toFixed(digits)
  if (Number(abs) === 0) return `${abs}%`
  return `${n > 0 ? '+' : '−'}${abs}%`
}

function localIso(d: Date): string {
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${d.getFullYear()}-${m}-${day}`
}

/**
 * "2026-11-15" → "Today" / "Tomorrow" / "Sun, Nov 15". The input is a calendar
 * day (the venue's game day), so it is parsed as a local date, never as UTC
 * midnight — `new Date('2026-11-15')` would roll back a day in US time zones.
 */
export function fmtDay(iso: string | null | undefined, opts?: { relative?: boolean }): string {
  if (!iso) return '—'
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso)
  if (!m) return iso
  const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]))
  if (opts?.relative !== false) {
    const now = new Date()
    if (localIso(now) === localIso(d)) return 'Today'
    const tomorrow = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1)
    if (localIso(tomorrow) === localIso(d)) return 'Tomorrow'
    const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1)
    if (localIso(yesterday) === localIso(d)) return 'Yesterday'
  }
  return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
}

/** "just now" / "4m ago" / "2h 10m ago" for a past epoch-ms timestamp. */
export function fmtAgo(ts: number, now = Date.now()): string {
  const s = Math.max(0, Math.round((now - ts) / 1000))
  if (s < 45) return 'just now'
  const m = Math.round(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m ago`
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**
 * Restore team casing on an outcome label. The server builds labels from the
 * canonical (lowercased) team key, so "North texas -20.5" / "Umkc +7.5" come
 * back sentence-cased. The event title carries the real spelling, so when the
 * label starts with one of the title's teams, borrow the title's casing
 * ("UMKC +7.5"). Otherwise capitalise each all-lowercase word.
 */
export function prettyOutcome(label: string, eventTitle?: string | null): string {
  if (!label) return label
  if (eventTitle) {
    const sides = eventTitle.split(/\s+(?:vs\.?|@|at)\s+/i).map(s => s.trim()).filter(Boolean)
    // Longest side first so "South Alabama" wins over a shorter prefix match.
    for (const side of sides.sort((a, b) => b.length - a.length)) {
      const re = new RegExp(`^${escapeRe(side)}(?=$|[\\s:])`, 'i')
      if (re.test(label)) return label.replace(re, side)
    }
  }
  return label.replace(/(^|\s)([a-z])(?=[a-z'.-]*)/g, (_m, sp: string, c: string) => sp + c.toUpperCase())
}
