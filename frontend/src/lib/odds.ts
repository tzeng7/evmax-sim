export function probToAmerican(p: number): string {
  if (p <= 0 || p >= 1) return '+100'
  if (p < 0.5) return '+' + Math.round((1 / p - 1) * 100)
  return '-' + Math.round((p / (1 - p)) * 100)
}

export function americanToProb(s: string): number | null {
  s = s.toString().trim()
  const n = parseFloat(s.replace('+', ''))
  if (isNaN(n) || n === 0) return null
  if (n > 0) return 100 / (n + 100)
  return Math.abs(n) / (Math.abs(n) + 100)
}

export function fmtDate(): string {
  return new Date().toISOString().slice(0, 10)
}

export function fmtTomorrow(): string {
  const d = new Date()
  d.setDate(d.getDate() + 1)
  return d.toISOString().slice(0, 10)
}

export function yesterday(): string {
  return new Date(Date.now() - 86400000).toISOString().slice(0, 10)
}
