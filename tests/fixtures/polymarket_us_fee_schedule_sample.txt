> ## Documentation Index
> Fetch the complete documentation index at: https://docs.polymarket.us/llms.txt
> Use this file to discover all available pages before exploring further.

# Fee Schedule

> Trading fee schedule, rebates, and examples

<Info>Effective exchange-wide from 10 AM ET, Thursday October 1, 2026.</Info>

## Standard Trading Fees

For the combo taker fee schedule, see [Combo Taker Fees](#combo-taker-fees).

Standard taker fees and maker rebates are computed using a symmetric formula that scales with price uncertainty:

```
Fee = Θ × C × p × (1 - p)
```

Where:

* **C** is the number of contracts
* **p** is the trade price (\$0.01 to \$0.99)
* **Θ** (theta) is the fee coefficient

| | Theta | Max (p = \$0.50) |
| - | - | - |
| **Taker Fee** | 0.0695 | \$1.74 |
| **Maker Rebate** | -0.0125 | -\$0.31 |

* **Maker rebate** is applied at the point of trade.
* **Taker rebate**: Participants who trade over \$250,000 in taker volume during the prior calendar month receive rebates according to the following schedule. A Participant's tier for a given month is determined by their taker volume in the immediately preceding calendar month. Rebates are paid out weekly.

| Prior calendar-month taker volume | % Taker fee rebate |
| - | - |
| \$250,000 - \$999,999 | 10% |
| \$1,000,000 - \$9,999,999 | 25% |
| \$10,000,000+ | 50% |

**How taker volume is measured**: Taker volume is the amount you put at risk on each fill where you are the taker. It is not the face value of the contracts (C × \$1.00).

```
Taker volume (buy)  = C × p
Taker volume (sell) = C × (1 - p)
```

Where **C** is the number of contracts filled and **p** is the trade price. Only fills where you are the taker count. Maker fills do not count. Each fill counts toward the calendar month of its execution time in Eastern Time (ET).

| Taker fill | Taker volume | Face value (not used) |
| - | - | - |
| Buy 1,000 contracts at \$0.10 | \$100 | \$1,000 |
| Sell 1,000 contracts at \$0.90 | \$100 | \$1,000 |
| Sell 1,000 contracts at \$0.30 | \$700 | \$1,000 |

**Accelerated Tier Placement**: A Participant may provide verifiable proof of their trailing-30-day notional trading volume on another prediction market and be assigned to the rebate tier corresponding to that volume.

<Note>
  API integrators: `C` is the number of **contracts** and `p` the **decimal** price. Execution reports carry these as fixed-point integers, and the collected fee in scaled notional units — see [Fees on execution reports](/partners/orders/data-model#fees-on-execution-reports) for how to decode `commission_notional_collected` with `price_scale` and `fractional_quantity_scale`.
</Note>

### Standard Fee Schedule by Price

| Price | Trade Value (100-lot) | Taker Pays (100-lot) | Maker Receives (100-lot) |
| - | - | - | - |
| \$0.01 | \$1 | \$0.07 | \$0.01 |
| \$0.02 | \$2 | \$0.14 | \$0.02 |
| \$0.03 | \$3 | \$0.20 | \$0.04 |
| \$0.04 | \$4 | \$0.27 | \$0.05 |
| \$0.05 | \$5 | \$0.33 | \$0.06 |
| \$0.06 | \$6 | \$0.39 | \$0.07 |
| \$0.07 | \$7 | \$0.45 | \$0.08 |
| \$0.08 | \$8 | \$0.51 | \$0.09 |
| \$0.09 | \$9 | \$0.57 | \$0.10 |
| \$0.10 | \$10 | \$0.63 | \$0.11 |
| \$0.11 | \$11 | \$0.68 | \$0.12 |
| \$0.12 | \$12 | \$0.73 | \$0.13 |
| \$0.13 | \$13 | \$0.79 | \$0.14 |
| \$0.14 | \$14 | \$0.84 | \$0.15 |
| \$0.15 | \$15 | \$0.89 | \$0.16 |
| \$0.16 | \$16 | \$0.93 | \$0.17 |
| \$0.17 | \$17 | \$0.98 | \$0.18 |
| \$0.18 | \$18 | \$1.03 | \$0.18 |
| \$0.19 | \$19 | \$1.07 | \$0.19 |
| \$0.20 | \$20 | \$1.11 | \$0.20 |
| \$0.21 | \$21 | \$1.15 | \$0.21 |
| \$0.22 | \$22 | \$1.19 | \$0.21 |
| \$0.23 | \$23 | \$1.23 | \$0.22 |
| \$0.24 | \$24 | \$1.27 | \$0.23 |
| \$0.25 | \$25 | \$1.30 | \$0.23 |
| \$0.26 | \$26 | \$1.34 | \$0.24 |
| \$0.27 | \$27 | \$1.37 | \$0.25 |
| \$0.28 | \$28 | \$1.40 | \$0.25 |
| \$0.29 | \$29 | \$1.43 | \$0.26 |
| \$0.30 | \$30 | \$1.46 | \$0.26 |
| \$0.31 | \$31 | \$1.49 | \$0.27 |
| \$0.32 | \$32 | \$1.51 | \$0.27 |
| \$0.33 | \$33 | \$1.54 | \$0.28 |
| \$0.34 | \$34 | \$1.56 | \$0.28 |
| \$0.35 | \$35 | \$1.58 | \$0.28 |
| \$0.36 | \$36 | \$1.60 | \$0.29 |
| \$0.37 | \$37 | \$1.62 | \$0.29 |
| \$0.38 | \$38 | \$1.64 | \$0.29 |
| \$0.39 | \$39 | \$1.65 | \$0.30 |
| \$0.40 | \$40 | \$1.67 | \$0.30 |
| \$0.41 | \$41 | \$1.68 | \$0.30 |
| \$0.42 | \$42 | \$1.69 | \$0.30 |
| \$0.43 | \$43 | \$1.70 | \$0.31 |
| \$0.44 | \$44 | \$1.71 | \$0.31 |
| \$0.45 | \$45 | \$1.72 | \$0.31 |
| \$0.46 | \$46 | \$1.73 | \$0.31 |
| \$0.47 | \$47 | \$1.73 | \$0.31 |
| \$0.48 | \$48 | \$1.73 | \$0.31 |
| \$0.49 | \$49 | \$1.74 | \$0.31 |
| \$0.50 | \$50 | \$1.74 | \$0.31 |
| \$0.51 | \$51 | \$1.74 | \$0.31 |
| \$0.52 | \$52 | \$1.73 | \$0.31 |
| \$0.53 | \$53 | \$1.73 | \$0.31 |
| \$0.54 | \$54 | \$1.73 | \$0.31 |
| \$0.55 | \$55 | \$1.72 | \$0.31 |
| \$0.56 | \$56 | \$1.71 | \$0.31 |
| \$0.57 | \$57 | \$1.70 | \$0.31 |
| \$0.58 | \$58 | \$1.69 | \$0.30 |
| \$0.59 | \$59 | \$1.68 | \$0.30 |
| \$0.60 | \$60 | \$1.67 | \$0.30 |
| \$0.61 | \$61 | \$1.65 | \$0.30 |
| \$0.62 | \$62 | \$1.64 | \$0.29 |
| \$0.63 | \$63 | \$1.62 | \$0.29 |
| \$0.64 | \$64 | \$1.60 | \$0.29 |
| \$0.65 | \$65 | \$1.58 | \$0.28 |
| \$0.66 | \$66 | \$1.56 | \$0.28 |
| \$0.67 | \$67 | \$1.54 | \$0.28 |
| \$0.68 | \$68 | \$1.51 | \$0.27 |
| \$0.69 | \$69 | \$1.49 | \$0.27 |
| \$0.70 | \$70 | \$1.46 | \$0.26 |
| \$0.71 | \$71 | \$1.43 | \$0.26 |
| \$0.72 | \$72 | \$1.40 | \$0.25 |
| \$0.73 | \$73 | \$1.37 | \$0.25 |
| \$0.74 | \$74 | \$1.34 | \$0.24 |
| \$0.75 | \$75 | \$1.30 | \$0.23 |
| \$0.76 | \$76 | \$1.27 | \$0.23 |
| \$0.77 | \$77 | \$1.23 | \$0.22 |
| \$0.78 | \$78 | \$1.19 | \$0.21 |
| \$0.79 | \$79 | \$1.15 | \$0.21 |
| \$0.80 | \$80 | \$1.11 | \$0.20 |
| \$0.81 | \$81 | \$1.07 | \$0.19 |
| \$0.82 | \$82 | \$1.03 | \$0.18 |
| \$0.83 | \$83 | \$0.98 | \$0.18 |
| \$0.84 | \$84 | \$0.93 | \$0.17 |
| \$0.85 | \$85 | \$0.89 | \$0.16 |
| \$0.86 | \$86 | \$0.84 | \$0.15 |
| \$0.87 | \$87 | \$0.79 | \$0.14 |
| \$0.88 | \$88 | \$0.73 | \$0.13 |
| \$0.89 | \$89 | \$0.68 | \$0.12 |
| \$0.90 | \$90 | \$0.63 | \$0.11 |
| \$0.91 | \$91 | \$0.57 | \$0.10 |
| \$0.92 | \$92 | \$0.51 | \$0.09 |
| \$0.93 | \$93 | \$0.45 | \$0.08 |
| \$0.94 | \$94 | \$0.39 | \$0.07 |
| \$0.95 | \$95 | \$0.33 | \$0.06 |
| \$0.96 | \$96 | \$0.27 | \$0.05 |
| \$0.97 | \$97 | \$0.20 | \$0.04 |
| \$0.98 | \$98 | \$0.14 | \$0.02 |
| \$0.99 | \$99 | \$0.07 | \$0.01 |

### Standard Fee Rules

* Standard taker fees and maker rebates are symmetric around p = 0.50 and lowest near the extremes (0 and 1).
* All fees and rebates are rounded to the nearest \$0.01 using banker's rounding (round half to even).
* When an aggressive order fills against multiple resting orders, each fill is charged its banker's-rounded fee, adjusted so that the total commission collected across the order's fills never exceeds the banker's rounding of the cumulative exact fee. The adjustment can only reduce a fill's charge, never increase it. Maker rebates are computed per fill, independently.

### Standard Fee Examples

#### Example 1: Buy 1,000 contracts at \$0.10 — cheap contract

Buying a long shot. The fee scales with price uncertainty: p × (1 − p) = 0.10 × 0.90 = 0.09.

* **Buyer (taker):** 0.0695 × 1,000 × 0.10 × 0.90 = **−\$6.26**
* **Seller (maker):** 0.0125 × 1,000 × 0.10 × 0.90 = **+\$1.12**

***

#### Example 2: Buy 1,000 contracts at \$0.65 — expensive contract

Buying a likely outcome. Higher price but lower p × (1 − p) than midpoint.

* **Buyer (taker):** 0.0695 × 1,000 × 0.65 × 0.35 = **−\$15.81**
* **Seller (maker):** 0.0125 × 1,000 × 0.65 × 0.35 = **+\$2.84**

***

#### Example 3: Sell 1,000 contracts at \$0.30 — sell low probability

The seller is the aggressor. Both sides pay based on the same p × (1 − p) factor.

* **Seller (taker):** 0.0695 × 1,000 × 0.30 × 0.70 = **−\$14.60**
* **Buyer (maker):** 0.0125 × 1,000 × 0.30 × 0.70 = **+\$2.62**

***

#### Example 4: Sell 1,000 contracts at \$0.90 — sell high probability

When the price is close to \$1.00, p × (1 − p) is small and fees are minimal.

* **Seller (taker):** 0.0695 × 1,000 × 0.90 × 0.10 = **−\$6.26**
* **Buyer (maker):** 0.0125 × 1,000 × 0.90 × 0.10 = **+\$1.12**

***

#### Example 5: Buy 1,000 contracts at \$0.50 — coin flip market

A 50/50 market. This is where the fee is highest per contract because p × (1 − p) = 0.25.

* **Buyer (taker):** 0.0695 × 1,000 × 0.50 × 0.50 = **−\$17.38**
* **Seller (maker):** 0.0125 × 1,000 × 0.50 × 0.50 = **+\$3.12**

## Combo Taker Fees

The taker side of a combo trade uses a separate fee curve:

```
Fee = C × p × [0.0695 × (1 - p) + 0.06 × (1 - p)^4]
```

`C` is the number of contracts and `p` is the combo execution price in decimal dollars. The curve applies to the combo execution as a whole, not separately to its component legs.

Maker rebates on combo fills continue to use the `-0.0125 × C × p × (1 - p)` formula above. A participant's taker rebate percentage is applied to the actual combo taker fee collected. Taker rebate tiers and payout cadence are unchanged.

### Combo Fee Schedule by Price

| Price | Trade Value (100-lot) | Combo Taker Pays (100-lot) |
| - | - | - |
| \$0.01 | \$1 | \$0.13 |
| \$0.02 | \$2 | \$0.25 |
| \$0.03 | \$3 | \$0.36 |
| \$0.04 | \$4 | \$0.47 |
| \$0.05 | \$5 | \$0.57 |
| \$0.06 | \$6 | \$0.67 |
| \$0.07 | \$7 | \$0.77 |
| \$0.08 | \$8 | \$0.86 |
| \$0.09 | \$9 | \$0.94 |
| \$0.10 | \$10 | \$1.02 |
| \$0.11 | \$11 | \$1.09 |
| \$0.12 | \$12 | \$1.17 |
| \$0.13 | \$13 | \$1.23 |
| \$0.14 | \$14 | \$1.30 |
| \$0.15 | \$15 | \$1.36 |
| \$0.16 | \$16 | \$1.41 |
| \$0.17 | \$17 | \$1.46 |
| \$0.18 | \$18 | \$1.51 |
| \$0.19 | \$19 | \$1.56 |
| \$0.20 | \$20 | \$1.60 |
| \$0.21 | \$21 | \$1.64 |
| \$0.22 | \$22 | \$1.68 |
| \$0.23 | \$23 | \$1.72 |
| \$0.24 | \$24 | \$1.75 |
| \$0.25 | \$25 | \$1.78 |
| \$0.26 | \$26 | \$1.80 |
| \$0.27 | \$27 | \$1.83 |
| \$0.28 | \$28 | \$1.85 |
| \$0.29 | \$29 | \$1.87 |
| \$0.30 | \$30 | \$1.89 |
| \$0.31 | \$31 | \$1.91 |
| \$0.32 | \$32 | \$1.92 |
| \$0.33 | \$33 | \$1.94 |
| \$0.34 | \$34 | \$1.95 |
| \$0.35 | \$35 | \$1.96 |
| \$0.36 | \$36 | \$1.96 |
| \$0.37 | \$37 | \$1.97 |
| \$0.38 | \$38 | \$1.97 |
| \$0.39 | \$39 | \$1.98 |
| \$0.40 | \$40 | \$1.98 |
| \$0.41 | \$41 | \$1.98 |
| \$0.42 | \$42 | \$1.98 |
| \$0.43 | \$43 | \$1.98 |
| \$0.44 | \$44 | \$1.97 |
| \$0.45 | \$45 | \$1.97 |
| \$0.46 | \$46 | \$1.96 |
| \$0.47 | \$47 | \$1.95 |
| \$0.48 | \$48 | \$1.95 |
| \$0.49 | \$49 | \$1.94 |
| \$0.50 | \$50 | \$1.92 |
| \$0.51 | \$51 | \$1.91 |
| \$0.52 | \$52 | \$1.90 |
| \$0.53 | \$53 | \$1.89 |
| \$0.54 | \$54 | \$1.87 |
| \$0.55 | \$55 | \$1.86 |
| \$0.56 | \$56 | \$1.84 |
| \$0.57 | \$57 | \$1.82 |
| \$0.58 | \$58 | \$1.80 |
| \$0.59 | \$59 | \$1.78 |
| \$0.60 | \$60 | \$1.76 |
| \$0.61 | \$61 | \$1.74 |
| \$0.62 | \$62 | \$1.71 |
| \$0.63 | \$63 | \$1.69 |
| \$0.64 | \$64 | \$1.67 |
| \$0.65 | \$65 | \$1.64 |
| \$0.66 | \$66 | \$1.61 |
| \$0.67 | \$67 | \$1.58 |
| \$0.68 | \$68 | \$1.56 |
| \$0.69 | \$69 | \$1.52 |
| \$0.70 | \$70 | \$1.49 |
| \$0.71 | \$71 | \$1.46 |
| \$0.72 | \$72 | \$1.43 |
| \$0.73 | \$73 | \$1.39 |
| \$0.74 | \$74 | \$1.36 |
| \$0.75 | \$75 | \$1.32 |
| \$0.76 | \$76 | \$1.28 |
| \$0.77 | \$77 | \$1.24 |
| \$0.78 | \$78 | \$1.20 |
| \$0.79 | \$79 | \$1.16 |
| \$0.80 | \$80 | \$1.12 |
| \$0.81 | \$81 | \$1.08 |
| \$0.82 | \$82 | \$1.03 |
| \$0.83 | \$83 | \$0.98 |
| \$0.84 | \$84 | \$0.94 |
| \$0.85 | \$85 | \$0.89 |
| \$0.86 | \$86 | \$0.84 |
| \$0.87 | \$87 | \$0.79 |
| \$0.88 | \$88 | \$0.74 |
| \$0.89 | \$89 | \$0.68 |
| \$0.90 | \$90 | \$0.63 |
| \$0.91 | \$91 | \$0.57 |
| \$0.92 | \$92 | \$0.51 |
| \$0.93 | \$93 | \$0.45 |
| \$0.94 | \$94 | \$0.39 |
| \$0.95 | \$95 | \$0.33 |
| \$0.96 | \$96 | \$0.27 |
| \$0.97 | \$97 | \$0.20 |
| \$0.98 | \$98 | \$0.14 |
| \$0.99 | \$99 | \$0.07 |

### Combo Fee Examples

#### Example 1: Buy 1,000 combo contracts at \$0.10 — low-priced combo

For a single fill, the combo fee is calculated from the combo execution price and quantity as a whole.

* **Buyer (taker):** 1,000 × 0.10 × \[0.0695 × 0.90 + 0.06 × 0.90^4] = \$10.1916 → **−\$10.19**
* **Seller (maker):** 0.0125 × 1,000 × 0.10 × 0.90 = **+\$1.12**

***

#### Example 2: Buy 1,000 combo contracts at \$0.50 — midpoint combo

At the midpoint, the combo taker fee is \$19.25.

* **Buyer (taker):** 1,000 × 0.50 × \[0.0695 × 0.50 + 0.06 × 0.50^4] = \$19.25 → **−\$19.25**
* **Seller (maker):** 0.0125 × 1,000 × 0.50 × 0.50 = **+\$3.12**

## FAQ

### Are fees deducted from my balance automatically?

Yes. Taker fees are deducted from your balance at the time of the trade. Maker rebates are credited to your balance at the time of the fill.

### Can fees ever be zero?

Yes. Fees are rounded to the nearest cent. On small trades (low quantity or prices near \$0.00 or \$1.00), the fee can round down to \$0.00.

### Do I pay fees when my order is canceled or expires?

No. Fees are only charged when a trade executes. If your order is canceled, expires, or is rejected, no fee is charged.

### What is banker's rounding?

Fees are rounded to the nearest cent using banker's rounding (round half to even). For example, \$0.025 rounds to \$0.02 (down to even), while \$0.035 rounds to \$0.04 (up to even).


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.