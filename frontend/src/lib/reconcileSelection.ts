// Pure reconcile of per-row scan UI state when the gap list changes.
// Keyed by market_id: state for ids still present is kept (selection, venue
// line-shop choice, hand-edited fills), defaults are seeded only for ids not
// seen before, and state for vanished ids is dropped.

export interface FillState { odds: string; stake: string }

export interface ReconcileGap {
  market_id: string
  mode?: string
}

export interface RowState {
  selected: Set<string>
  venueSel: Record<string, string>
  fills: Record<string, FillState>
}

export function reconcileRowState<G extends ReconcileGap>(
  prev: RowState,
  prevIds: ReadonlySet<string>,
  gaps: readonly G[],
  legsOf: (g: G) => readonly G[],
  defaultFill: (leg: G) => FillState,
): { state: RowState; ids: Set<string> } {
  const ids = new Set<string>()
  const selected = new Set<string>()
  const venueSel: Record<string, string> = {}
  const fills: Record<string, FillState> = {}
  for (const g of gaps) {
    const mid = g.market_id
    ids.add(mid)
    if (prevIds.has(mid)) {
      if (prev.selected.has(mid)) selected.add(mid)
    } else if ((g.mode || 'live') === 'live') {
      selected.add(mid)
    }
    if (mid in prev.venueSel) venueSel[mid] = prev.venueSel[mid]
    for (const leg of legsOf(g)) {
      fills[leg.market_id] = prev.fills[leg.market_id] ?? defaultFill(leg)
    }
  }
  return { state: { selected, venueSel, fills }, ids }
}
