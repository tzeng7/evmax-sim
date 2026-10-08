"""EV calculation: devigging, probability → edge, Kelly sizing.

- devig.py      — Power Method devigging (2-way and 3-way markets) via scipy.brentq
- calculator.py — EV = (true_prob × payout) - 1; evaluates YES side only
- kelly.py      — Fractional Kelly with liquidity discount, hard cap at 5% of bankroll
- maker_pilot.py — Maker-only ¼-Kelly pilot for shallow NFL underdog spread_pmf rungs; per-game cap; checkpoint verdict
"""
