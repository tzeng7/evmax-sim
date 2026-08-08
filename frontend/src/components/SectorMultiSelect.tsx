import { useEffect, useMemo, useRef, useState } from 'react'
import { useCategories } from '../lib/useCategories'

interface Props {
  /** Comma-separated list of selected sector keys (e.g. "nba,nfl"). */
  value: string
  onChange: (sectors: string) => void
  /** Toggle label when nothing is checked. The arb page passes "Default
   *  sectors" because its backend treats an empty selection as "every
   *  sector both venues carry", not "scan nothing". */
  emptyLabel?: string
}

/**
 * Sector picker for the scan bar — a command-menu-style checklist. The user can
 * search, toggle "All Sectors", check individual sectors, and clear. The
 * selection is still emitted as a comma-separated string of keys to keep the
 * /api/scan contract unchanged.
 */
export function SectorMultiSelect({ value, onChange, emptyLabel = 'No sectors' }: Props) {
  const cats = useCategories()
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const ref = useRef<HTMLDivElement>(null)
  const searchRef = useRef<HTMLInputElement>(null)

  const selected = new Set(value ? value.split(',').filter(Boolean) : [])
  const allKeys = cats.map(c => c.key)
  const allSelected = allKeys.length > 0 && allKeys.every(k => selected.has(k))

  // Close on outside click or Escape (clearing the query as we close), and
  // focus the search when opening. Query reset lives in the close handlers,
  // not the effect body, to avoid a synchronous setState-in-effect.
  useEffect(() => {
    if (!open) return
    const close = () => { setOpen(false); setQuery('') }
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) close()
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    // Focus after the open transition begins so the caret lands in the field.
    const t = setTimeout(() => searchRef.current?.focus(), 20)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
      clearTimeout(t)
    }
  }, [open])

  const handleToggle = () => {
    if (open) { setOpen(false); setQuery('') }
    else setOpen(true)
  }

  const emit = (next: Set<string>) => {
    // Preserve registry order so the string is stable regardless of click order.
    onChange(allKeys.filter(k => next.has(k)).join(','))
  }

  const toggle = (key: string) => {
    const next = new Set(selected)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    emit(next)
  }

  const toggleAll = () => emit(allSelected ? new Set() : new Set(allKeys))

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return cats
    return cats.filter(c =>
      c.key.toLowerCase().includes(q) || (c.display_name || '').toLowerCase().includes(q),
    )
  }, [cats, query])

  const label = allSelected
    ? 'All Sectors'
    : selected.size === 0
      ? emptyLabel
      : selected.size === 1
        ? [...selected][0].toUpperCase()
        : `${selected.size} sectors`

  return (
    <div className="sector-ms" ref={ref}>
      <button
        type="button"
        className={`sector-ms-toggle ${open ? 'open' : ''}`}
        onClick={handleToggle}
        title="Choose one or more sectors to scan"
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        <span className="sector-ms-label">{label}</span>
        <svg className="sector-ms-caret" width="12" height="12" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>
      {/* Kept mounted so the exit animation can play; `visibility: hidden` when
          closed keeps the controls out of the tab order and a11y tree. */}
      <div className={`sector-ms-menu ${open ? 'open' : ''}`} role="listbox" aria-multiselectable="true">
        <div className="sector-ms-search">
          <input
            ref={searchRef}
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search sectors…"
            aria-label="Search sectors"
          />
        </div>
        <div className="sector-ms-list">
          {!query && (
            <>
              <label className="sector-ms-item sector-ms-all">
                <input type="checkbox" checked={allSelected} onChange={toggleAll} />
                <span>All Sectors</span>
              </label>
              <div className="sector-ms-divider" />
            </>
          )}
          {filtered.length === 0 && <div className="sector-ms-none">No matches</div>}
          {filtered.map(c => (
            <label key={c.key} className="sector-ms-item" title={c.display_name}>
              <input type="checkbox" checked={selected.has(c.key)} onChange={() => toggle(c.key)} />
              <span>{c.key.toUpperCase()}</span>
            </label>
          ))}
        </div>
        <div className="sector-ms-footer">
          <span>{selected.size} selected</span>
          <button type="button" className="sector-ms-clear" onClick={() => onChange('')} disabled={selected.size === 0}>
            Clear
          </button>
        </div>
      </div>
    </div>
  )
}
