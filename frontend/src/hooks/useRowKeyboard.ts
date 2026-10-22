import { useCallback, useEffect, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'

/** True when the event target is somewhere the user is typing. */
export function isTypingTarget(t: EventTarget | null): boolean {
  const el = t as HTMLElement | null
  if (!el || !el.tagName) return false
  const tag = el.tagName
  if (tag === 'SELECT' || tag === 'TEXTAREA' || el.isContentEditable) return true
  if (tag === 'INPUT') {
    const type = (el as HTMLInputElement).type
    return type !== 'checkbox' && type !== 'radio' && type !== 'button'
  }
  return false
}

interface Options {
  /** Row ids in display order. */
  ids: string[]
  onToggle: (id: string) => void
  /** Triggered by `p`. */
  onPick?: () => void
}

/**
 * Keyboard navigation for a table: j/k or ArrowDown/ArrowUp move the row
 * focus, Space toggles the focused row's selection, `p` runs onPick.
 * Spread `containerProps` on the scrollable wrapper (it becomes a tab stop)
 * and `rowProps(id)` on each <tr>. Keys are ignored while typing in a
 * text input / select, and when a modifier key is held.
 */
export function useRowKeyboard({ ids, onToggle, onPick }: Options) {
  const [rawFocusedId, setFocusedId] = useState<string | null>(null)
  // A row that vanished (filter change, re-scan) counts as unfocused.
  const focusedId = rawFocusedId && ids.includes(rawFocusedId) ? rawFocusedId : null
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!focusedId) return
    const rows = containerRef.current?.querySelectorAll<HTMLElement>('tr[data-rowid]')
    rows?.forEach(r => { if (r.dataset.rowid === focusedId) r.scrollIntoView({ block: 'nearest' }) })
  }, [focusedId])

  const onKeyDown = useCallback((e: KeyboardEvent<HTMLElement>) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return
    if (isTypingTarget(e.target)) return
    const move = (delta: number) => {
      if (!ids.length) return
      e.preventDefault()
      const i = focusedId ? ids.indexOf(focusedId) : -1
      const next = i < 0 ? (delta > 0 ? 0 : ids.length - 1) : Math.min(ids.length - 1, Math.max(0, i + delta))
      setFocusedId(ids[next])
    }
    if (e.key === 'j' || e.key === 'ArrowDown') move(1)
    else if (e.key === 'k' || e.key === 'ArrowUp') move(-1)
    else if (e.key === ' ' && focusedId) {
      // Space on a focused checkbox/button already activates it natively.
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'BUTTON') return
      e.preventDefault()
      onToggle(focusedId)
    } else if (e.key === 'p' && onPick) {
      e.preventDefault()
      onPick()
    }
  }, [ids, focusedId, onToggle, onPick])

  const containerProps = {
    ref: containerRef,
    tabIndex: 0,
    onKeyDown,
    className: 'table-scroll kb-table',
    'aria-label': 'Table. j and k move between rows, space toggles selection, p picks selected. Press ? for all shortcuts.',
  }
  const rowProps = (id: string) => ({
    'data-rowid': id,
    className: id === focusedId ? 'kb-focus' : undefined,
    'aria-current': id === focusedId ? ('true' as const) : undefined,
    onClick: () => setFocusedId(id),
  })
  return { containerProps, rowProps, focusedId }
}
