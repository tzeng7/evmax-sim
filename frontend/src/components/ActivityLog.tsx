import { useState } from 'react'
import { activityAsText, clearActivity, dismissActivity, markActivitySeen, type ActivityEntry } from '../hooks/activityLog'

export function ActivityLog({ entries }: { entries: ActivityEntry[] }) {
  const [open, setOpen] = useState(false)
  const unreadErr = entries.some((e) => e.type === 'err' && !e.seen)

  const toggle = () => {
    if (!open) markActivitySeen()
    setOpen(!open)
  }
  const copy = (list: ActivityEntry[]) => {
    navigator.clipboard?.writeText(activityAsText(list)).catch(() => {})
  }
  const ordered = [...entries].reverse()

  return (
    <>
      <button type="button" className="btn activity-btn" onClick={toggle} aria-expanded={open}>
        Activity
        {unreadErr && <span className="activity-dot" aria-label="Unread errors" />}
      </button>
      {open && (
        <aside className="activity-drawer" role="dialog" aria-label="Activity log">
          <div className="activity-head">
            <strong>Activity ({entries.length})</strong>
            <span className="activity-head-actions">
              <button type="button" className="btn" disabled={!entries.length} onClick={() => copy(ordered)}>Copy all</button>
              <button type="button" className="btn" disabled={!entries.length} onClick={clearActivity}>Clear</button>
              <button type="button" className="btn" onClick={() => setOpen(false)}>Close</button>
            </span>
          </div>
          {ordered.length === 0 && <div className="activity-empty">Nothing logged this session.</div>}
          <ul className="activity-list">
            {ordered.map((e) => (
              <li key={e.id} className={`activity-item ${e.type}`}>
                <div className="activity-meta">
                  <span>{new Date(e.at).toLocaleTimeString()}</span>
                  <span className="activity-type">{e.type === 'err' ? 'error' : e.type}</span>
                  <button type="button" className="activity-link" onClick={() => copy([e])}>copy</button>
                  <button type="button" className="activity-link" onClick={() => dismissActivity(e.id)}>dismiss</button>
                </div>
                <div className="activity-msg">{e.message}</div>
                {e.detail && <pre className="activity-detail">{e.detail}</pre>}
              </li>
            ))}
          </ul>
        </aside>
      )}
    </>
  )
}
