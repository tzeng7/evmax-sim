import { useCallback, useEffect, useState } from 'react'

export type Page =
  | { kind: 'dashboard' } | { kind: 'metrics' } | { kind: 'board' } | { kind: 'arb' }
  | { kind: 'projections' } | { kind: 'portfolios' } | { kind: 'portfolio'; id: string }
export type DashView = 'all' | 'placed'
export interface Route { page: Page; view: DashView }

const SIMPLE = ['dashboard', 'metrics', 'board', 'arb', 'projections', 'portfolios'] as const

export function parseHash(hash: string): Route {
  const [path, query = ''] = hash.replace(/^#\/?/, '').split('?')
  const view: DashView = new URLSearchParams(query).get('view') === 'placed' ? 'placed' : 'all'
  const [head, ...rest] = path.split('/')
  if (head === 'portfolio' && rest[0]) return { page: { kind: 'portfolio', id: decodeURIComponent(rest[0]) }, view }
  const kind = SIMPLE.find((k) => k === head)
  return { page: { kind: kind ?? 'dashboard' } as Page, view }
}

export function formatHash({ page, view }: Route): string {
  const path = page.kind === 'portfolio' ? `portfolio/${encodeURIComponent(page.id)}` : page.kind
  return `#/${path}${page.kind === 'dashboard' && view === 'placed' ? '?view=placed' : ''}`
}

/** Page + portfolio id + dashboard view live in location.hash, so reload and
 *  deep links restore them and back/forward walks between pages. */
export function useHashRoute(): [Route, (r: Partial<Route>) => void] {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash))
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash))
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  const update = useCallback((patch: Partial<Route>) => {
    const next = { ...parseHash(window.location.hash), ...patch }
    const hash = formatHash(next)
    if (hash !== window.location.hash) window.location.hash = hash // pushes history; fires hashchange
  }, [])
  return [route, update]
}
