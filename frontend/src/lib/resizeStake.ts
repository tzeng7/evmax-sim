// Shared stake re-size used by ScanResults and OpenPositions so the two views
// can't drift. The stored kelly_fraction is K_full × scan_kelly × discounts,
// capped at 5%, so scaling by (current_kelly / scan_kelly) recovers the size
// for a new Kelly knob without a rescan. Re-applies the 5% per-bet cap. The 8%
// per-event exposure guard isn't reproduced — for canonical sizing, rescan.
export const STAKE_CAP = 0.05

export function resizeStake(storedKellyFraction: number, bankroll: number, kelly: number, scanKelly: number): number {
  const stored = storedKellyFraction || 0
  const scaled = scanKelly > 0 ? stored * (kelly / scanKelly) : stored
  return bankroll * Math.min(scaled, STAKE_CAP)
}
