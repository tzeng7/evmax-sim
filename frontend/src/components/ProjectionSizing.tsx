import type { ProjectionGame, ProjectionSizing } from '../lib/types'
import { betable, breakeven, stakeFraction, validOdds, type StakeContext } from '../lib/projections'
import type { SizingSettings, useStakeContext } from '../lib/stakeSizing'

const KELLY_CHOICES: [number, string][] = [[1, 'Full'], [0.5, 'Half'], [0.25, 'Quarter']]

const money = (v: number) => `$${v >= 100 ? v.toFixed(0) : v.toFixed(2)}`

/** Every spread / O/U pick's stake in this slate under `ctx` (games not yet kicked off only). */
function slateStakes(games: ProjectionGame[], ctx: StakeContext): number[] {
  return games.filter(g => betable(g))
    .flatMap(g => [g.pick?.spread ? g.pick.spread_p : undefined, g.pick?.total ? g.pick.total_p : undefined])
    .filter((p): p is number => p != null)
    .map(p => ctx.bankroll * stakeFraction(p, ctx.odds, ctx.kelly, ctx.cap))
}

interface BarProps {
  sizing: ProjectionSizing
  games: ProjectionGame[]
  state: ReturnType<typeof useStakeContext>
}

/**
 * Stake sizing for the slate's model picks: Kelly on each pick's calibrated win
 * probability at the chosen price. Honest by construction — at -110 the picks'
 * calibrated rates sit below break-even, so the stake is $0.
 */
export function ProjectionSizingBar({ sizing, games, state }: BarProps) {
  const { settings, setSettings, ctx } = state
  const set = (patch: Partial<SizingSettings>) => setSettings({ ...settings, ...patch })
  const stakes = ctx ? slateStakes(games, ctx) : []
  const bets = stakes.filter(s => s > 0)
  const odds = ctx?.odds ?? (sizing.presets.find(p => p.key === settings.preset)?.odds ?? settings.customOdds)
  return (
    <div className="proj-sizing">
      <label className="proj-field" title="Your bankroll in dollars; stakes are a fraction of it">
        <span className="proj-field-label">Bankroll</span>
        <input
          type="number" inputMode="decimal" min={0} step={50} placeholder="$"
          value={settings.bankroll ?? ''}
          onChange={e => set({ bankroll: e.target.value === '' ? null : Number(e.target.value) })}
          style={{ width: 96 }}
        />
      </label>
      <label className="proj-field" title="Where you would bet the pick (exchange prices include the fee)">
        <span className="proj-field-label">Price</span>
        <select value={settings.preset} onChange={e => set({ preset: e.target.value })}>
          {sizing.presets.map(p => <option key={p.key} value={p.key}>{p.label}</option>)}
          <option value="custom">Custom odds</option>
        </select>
      </label>
      {settings.preset === 'custom' && (
        <label className="proj-field" title="American odds, e.g. -110 or +105">
          <span className="proj-field-label">Odds</span>
          <input
            type="number" inputMode="numeric" step={1} value={settings.customOdds}
            onChange={e => set({ customOdds: Number(e.target.value) })}
            style={{ width: 76 }}
          />
        </label>
      )}
      <label className="proj-field" title={`Fraction of full Kelly; one stake is capped at ${(sizing.cap * 100).toFixed(0)}% of bankroll`}>
        <span className="proj-field-label">Kelly</span>
        <select
          value={String(settings.kelly ?? sizing.default_kelly)}
          onChange={e => set({ kelly: Number(e.target.value) })}
        >
          {KELLY_CHOICES.map(([v, l]) => <option key={v} value={String(v)}>{l}</option>)}
        </select>
      </label>
      <span className="muted proj-sizing-summary" title={sizing.note}>
        {!validOdds(odds)
          ? 'Odds must be ≤ −100 or ≥ +100.'
          : !ctx
            ? `Enter a bankroll to size picks · break-even ${(breakeven(odds) * 100).toFixed(1)}%`
            : bets.length === 0
              ? `No bets at this price: no open pick's calibrated win rate beats the ${(breakeven(odds) * 100).toFixed(1)}% break-even`
              : `${bets.length} bet${bets.length === 1 ? '' : 's'} · ${money(bets.reduce((a, b) => a + b, 0))} total · break-even ${(breakeven(odds) * 100).toFixed(1)}%`}
      </span>
    </div>
  )
}

/** One pick's stake, with the calibrated probability vs break-even in the tooltip. */
export function StakeTag({ p, ctx }: { p: number | null | undefined; ctx: StakeContext }) {
  if (p == null) return null
  const f = stakeFraction(p, ctx.odds, ctx.kelly, ctx.cap)
  const be = breakeven(ctx.odds)
  const why = `calibrated ${(p * 100).toFixed(1)}% vs break-even ${(be * 100).toFixed(1)}% at ${ctx.oddsLabel}`
  if (f <= 0) return <span className="muted" title={`No bet: ${why}`}> · $0</span>
  return (
    <span className="green" title={`${(f * 100).toFixed(2)}% of bankroll: ${why}`}>
      {' '}· {money(ctx.bankroll * f)}
    </span>
  )
}
