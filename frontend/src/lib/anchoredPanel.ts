import { useLayoutEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'

export interface Rect { left: number; right: number; top: number; bottom: number }
export interface Size { width: number; height: number }
export interface Placement { left: number; top: number; width: number; maxHeight: number }
export interface PlaceOptions {
  /** 'end' lines the panel's right edge up with the anchor's; 'start' the left edges. */
  align?: 'start' | 'end'
  /** Space between the anchor and the panel (px). */
  gap?: number
  /** Minimum distance kept from every viewport edge (px). */
  margin?: number
}

/**
 * Where to put a popup panel so that it stays inside the viewport.
 *
 * The preferred spot is next to the anchor (below it, aligned per `align`).
 * The panel is then shifted horizontally to stay `margin` px inside the
 * viewport, and narrowed when the viewport is narrower than the panel. It opens
 * above the anchor only when it does not fit below and there is more room
 * above. `maxHeight` is the room left on that side; the panel scrolls beyond it.
 *
 * Pure: no DOM access, so it is unit-tested directly.
 */
export function placeAnchored(anchor: Rect, panel: Size, viewport: Size, opts: PlaceOptions = {}): Placement {
  const { align = 'end', gap = 8, margin = 16 } = opts
  const width = Math.max(0, Math.min(panel.width, viewport.width - 2 * margin))
  const preferred = align === 'end' ? anchor.right - width : anchor.left
  const left = Math.min(Math.max(preferred, margin), viewport.width - margin - width)

  const roomBelow = viewport.height - anchor.bottom - gap - margin
  const roomAbove = anchor.top - gap - margin
  if (panel.height <= roomBelow || roomBelow >= roomAbove) {
    return { left, top: anchor.bottom + gap, width, maxHeight: Math.max(0, roomBelow) }
  }
  const height = Math.min(panel.height, roomAbove)
  return { left, top: anchor.top - gap - height, width, maxHeight: Math.max(0, roomAbove) }
}

/**
 * Positions an open popup panel next to its anchor with `position: fixed`,
 * clamped inside the viewport by {@link placeAnchored}. The panel's CSS must cap
 * its width at the viewport (`max-width: calc(100vw - 32px)`). Fixed positioning also
 * escapes ancestors with `overflow: hidden|auto` (table scrollers, cards) that
 * clip an absolutely positioned panel.
 *
 * Re-places on window resize, on any scroll, and when the panel's own size
 * changes (content arriving after it opened). The panel is hidden until the
 * first placement so it never flashes at the wrong spot.
 */
export function useAnchoredPanel<A extends HTMLElement, P extends HTMLElement>(open: boolean, opts: PlaceOptions = {}) {
  const anchorRef = useRef<A>(null)
  const panelRef = useRef<P>(null)
  const [style, setStyle] = useState<CSSProperties>(HIDDEN)
  const { align = 'end', gap = 8, margin = 16 } = opts

  useLayoutEffect(() => {
    if (!open) return
    let last = ''
    const update = () => {
      const a = anchorRef.current
      const p = panelRef.current
      if (!a || !p) return
      // Width is capped by the panel's own CSS (max-width: calc(100vw - 32px)),
      // so offsetWidth is already viewport-safe. scrollHeight is the content
      // height whatever max-height is applied. Nothing here writes to the DOM,
      // so the ResizeObserver below cannot loop.
      const size = { width: p.offsetWidth, height: p.scrollHeight }
      const viewport = { width: document.documentElement.clientWidth, height: window.innerHeight }
      const pl = placeAnchored(a.getBoundingClientRect(), size, viewport, { align, gap, margin })
      const key = `${pl.left}|${pl.top}|${pl.maxHeight}`
      if (key === last) return
      last = key
      setStyle({ position: 'fixed', left: pl.left, top: pl.top, maxHeight: pl.maxHeight, overflow: 'auto' })
    }
    update()
    window.addEventListener('resize', update)
    window.addEventListener('scroll', update, true)
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(update) : null
    if (ro && panelRef.current) ro.observe(panelRef.current)
    return () => {
      window.removeEventListener('resize', update)
      window.removeEventListener('scroll', update, true)
      ro?.disconnect()
    }
  }, [open, align, gap, margin])

  // Layout effects run before paint, so a re-opened panel is placed before it
  // is ever shown at its previous spot.
  return { anchorRef, panelRef, style: open ? style : HIDDEN }
}

const HIDDEN: CSSProperties = { position: 'fixed', left: 0, top: 0, visibility: 'hidden' }
