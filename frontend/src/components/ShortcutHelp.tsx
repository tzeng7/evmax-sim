import { useEffect, useRef, useState } from 'react'
import { isTypingTarget } from '../hooks/useRowKeyboard'

const SHORTCUTS: [string, string][] = [
  ['Tab', 'Focus a table (then use the keys below)'],
  ['j / ↓', 'Next row'],
  ['k / ↑', 'Previous row'],
  ['Space', 'Toggle selection of the focused row'],
  ['p', 'Pick selected rows in the focused table'],
  ['?', 'Show / hide this help'],
  ['Esc', 'Close this help'],
]

export function ShortcutHelp() {
  const [open, setOpen] = useState(false)
  const closeRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
      else if (e.key === '?' && !e.ctrlKey && !e.metaKey && !e.altKey && !isTypingTarget(e.target)) {
        e.preventDefault()
        setOpen(o => !o)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => { if (open) closeRef.current?.focus() }, [open])

  if (!open) return null
  return (
    <div className="kb-overlay" onClick={() => setOpen(false)}>
      <div className="kb-dialog panel" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts"
        onClick={e => e.stopPropagation()}>
        <div className="panel-header">
          <h2>Keyboard shortcuts</h2>
          <button ref={closeRef} className="btn btn-sm" onClick={() => setOpen(false)} aria-label="Close shortcut help">Close</button>
        </div>
        <table>
          <tbody>
            {SHORTCUTS.map(([k, d]) => (
              <tr key={k}><td><kbd>{k}</kbd></td><td>{d}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
