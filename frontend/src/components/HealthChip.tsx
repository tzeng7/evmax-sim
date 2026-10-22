import { useCallback, useEffect, useRef, useState } from 'react'
import './HealthChip.css'

type Sev = 'ok' | 'info' | 'warning' | 'critical'
interface CheckIssue { severity: Sev; detail: string }
interface CheckRow { check: string; severity: Sev; issues: CheckIssue[] }
interface Integrity {
  status: 'ok' | 'warning' | 'critical'
  checks: CheckRow[]
  failed: string[]
  generated_at: string
}

const REFRESH_MS = 90_000

function ago(iso: string, now: number): string {
  const s = Math.max(0, Math.round((now - Date.parse(iso)) / 1000))
  if (Number.isNaN(s)) return '?'
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  return `${Math.floor(s / 3600)}h ago`
}

/**
 * Top-bar health chip over GET /api/integrity (the read-only `evmax cleanup
 * integrity` sweep). Green/amber/red = worst severity. Fails soft: any fetch
 * error shows "unknown" and never throws into the page.
 */
export function HealthChip() {
  const [data, setData] = useState<Integrity | null>(null)
  const [failed, setFailed] = useState(false)
  const [open, setOpen] = useState(false)
  const [now, setNow] = useState(() => Date.now())
  const [withPinnacle, setWithPinnacle] = useState(false)
  const root = useRef<HTMLDivElement>(null)

  const load = useCallback(async (pinnacle: boolean) => {
    try {
      const res = await fetch(`/api/integrity${pinnacle ? '?check_pinnacle=true' : ''}`)
      if (!res.ok) throw new Error(String(res.status))
      setData((await res.json()) as Integrity)
      setFailed(false)
    } catch {
      setFailed(true)
    }
    setNow(Date.now())
  }, [])

  useEffect(() => {
    void load(withPinnacle)
    const id = window.setInterval(() => void load(withPinnacle), REFRESH_MS)
    return () => window.clearInterval(id)
  }, [load, withPinnacle])

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const level = failed || !data ? 'unknown' : data.status === 'critical' ? 'red' : data.status === 'warning' ? 'amber' : 'green'
  const label = level === 'unknown' ? 'unknown' : level === 'red' ? 'critical' : level === 'amber' ? 'warning' : 'healthy'

  return (
    <div className="health-chip" ref={root}>
      <button
        type="button"
        className={`health-btn ${level}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        title="System integrity"
        onClick={() => { setOpen((o) => !o); setNow(Date.now()) }}
      >
        <span className="health-dot" aria-hidden="true" />
        {label}
      </button>
      {open && (
        <div className="health-panel" role="dialog" aria-label="Integrity checks">
          <div className="health-head">
            <strong>Integrity</strong>
            <span className="health-age">
              {data ? `checked ${ago(data.generated_at, now)}` : 'no data'}
            </span>
          </div>
          {failed && <div className="health-err">Integrity endpoint unavailable{data ? ' — showing last result' : ''}.</div>}
          <label className="health-opt">
            <input type="checkbox" checked={withPinnacle} onChange={(e) => setWithPinnacle(e.target.checked)} />
            Probe Pinnacle (live request)
          </label>
          <ul className="health-list">
            {data?.checks.map((c) => (
              <li key={c.check} className="health-row">
                <div className="health-row-top">
                  <span className={`health-sev ${c.severity}`}>{c.severity}</span>
                  <span className="health-check">{c.check}</span>
                  <span className="health-age">{ago(data.generated_at, now)}</span>
                </div>
                {c.issues.map((i, k) => (
                  <div key={k} className="health-msg">{i.detail}</div>
                ))}
              </li>
            ))}
          </ul>
          {data && data.failed.length > 0 && (
            <div className="health-err">Crashed checks: {data.failed.join(', ')}</div>
          )}
        </div>
      )}
    </div>
  )
}
