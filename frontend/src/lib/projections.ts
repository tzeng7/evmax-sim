import type { ProjectionOption, ProjectionOptionValues } from './types'

/** The declared defaults of an engine's options (null = auto). */
export function optionDefaults(specs: ProjectionOption[] | undefined): ProjectionOptionValues {
  return Object.fromEntries((specs ?? []).map(s => [s.key, s.default]))
}

/** True for valid American odds (<= -100 or >= +100). */
export function validOdds(odds: number): boolean {
  return Number.isFinite(odds) && (odds <= -100 || odds >= 100)
}

/** The win probability a bet at these American odds needs to break even (-110 -> 0.524). */
export function breakeven(odds: number): number {
  return odds < 0 ? -odds / (-odds + 100) : 100 / (odds + 100)
}

/**
 * Stake as a bankroll fraction: `kelly` x full Kelly at these odds, 0 when the bet is
 * not +EV, capped at `cap`. Mirrors evmax.nfl_projections.picks.stake_fraction.
 */
export function stakeFraction(p: number | null | undefined, odds: number, kelly: number, cap: number): number {
  if (p == null || !validOdds(odds)) return 0
  const be = breakeven(odds)
  return Math.min(cap, Math.max(0, (p - be) / (1 - be)) * kelly)
}

/** What a stake is sized against: bankroll in dollars, American odds, Kelly fraction and cap. */
export interface StakeContext {
  bankroll: number
  odds: number
  oddsLabel: string
  kelly: number
  cap: number
}

/** Whether a game can still be bet: no final score and (when known) kickoff still ahead. */
export function betable(g: { kickoff: string | null; actual_home: number | null }, now: number = Date.now()): boolean {
  if (g.actual_home != null) return false
  if (!g.kickoff) return true
  const t = new Date(g.kickoff).getTime()
  return isNaN(t) || t > now
}
