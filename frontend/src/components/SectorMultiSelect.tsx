import { useEffect, useRef, useState } from 'react'
import { useCategories } from '../lib/useCategories'

interface Props {
  /** Comma-separated list of selected sector keys (e.g. "nba,nfl"). */
  value: string
  onChange: (sectors: string) => void
}

/**
 * Sector picker for the scan bar. Replaces the old single-choice <select>
 * with a checkbox dropdown so the user can scan several specific sectors at
 * once. The selection is still emitted as a comma-separated string of keys to
 * keep the /api/scan contract unchanged.
 */
export function SectorMultiSelect({ value, onChange }: Props) {
  const cats = useCategories()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const selected = new Set(value ? value.split(',').filter(Boolean) : [])
  const allKeys = cats.map(c => c.key)
  const allSelected = allKeys.length > 0 && allKeys.every(k => selected.has(k))

  // Close the dropdown when clicking outside it.
  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [open])

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

  const toggleAll = () => {
    emit(allSelected ? new Set() : new Set(allKeys))
  }

  const label = allSelected
    ? 'All Sectors'
    : selected.size === 0
      ? 'No sectors'
      : selected.size === 1
        ? [...selected][0].toUpperCase()
        : `${selected.size} sectors`

  return (
    <div className="sector-ms" ref={ref}>
      <button
        type="button"
        className="sector-ms-toggle"
        onClick={() => setOpen(o => !o)}
        title="Choose one or more sectors to scan"
      >
        <span>{label}</span>
        <span className="sector-ms-caret">▾</span>
      </button>
      {open && (
        <div className="sector-ms-menu" role="listbox" aria-multiselectable="true">
          <label className="sector-ms-item sector-ms-all">
            <input type="checkbox" checked={allSelected} onChange={toggleAll} />
            <span>All Sectors</span>
          </label>
          <div className="sector-ms-divider" />
          {cats.map(c => (
            <label key={c.key} className="sector-ms-item" title={c.display_name}>
              <input type="checkbox" checked={selected.has(c.key)} onChange={() => toggle(c.key)} />
              <span>{c.key.toUpperCase()}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  )
}
