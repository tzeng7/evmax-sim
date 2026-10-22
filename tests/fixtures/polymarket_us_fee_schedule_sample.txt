> ## Documentation Index
> Fetch the complete documentation index at: https://docs.polymarket.us/llms.txt
> Use this file to discover all available pages before exploring further.

# Fee Schedule

> Trading fee schedule, rebates, and examples

> [!NOTE]
>
> Effective exchange-wide from 1:00 PM ET, Monday, October 19, 2026.

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

| | Theta | Max (100-lot at p = \$0.50) |
| - | - | - |
| **Taker Fee** | 0.0695 | \$1.74 |
| **Taker Fee (Table Tennis)** | 0.10 | \$2.50 |
| **Maker Rebate** | -0.0035 to -0.0104, by [maker tier](#maker-rebate-tiers) | -\$0.09 to -\$0.26 |

Table Tennis markets use a higher taker fee coefficient (Θ = 0.10). Maker rebates use the same tier coefficient in every market, including Table Tennis and crypto. The schedule by price below uses the standard 0.0695 coefficient. See Example 6 under [Standard Fee Examples](#standard-fee-examples) for a Table Tennis fee.

* **Maker rebate** is applied at the point of trade, at your maker tier's coefficient. Combo fills carry no maker rebate. See [Combo Taker Fees](#combo-taker-fees).
* **Taker rebate** is a percentage of the taker fees you paid, including combo taker fees, set by your taker tier. Rebates are paid out weekly.

### Maker rebate tiers

Your maker tier is set by your maker volume in the prior calendar month.

| Maker tier | Prior calendar-month maker volume | Maker rebate coefficient (Θ) | Share of the standard taker fee |
| - | - | - | - |
| Tier 1 | Below \$10,000,000 | No maker rebate | 0% |
| Tier 2 | \$10,000,000 - \$49,999,999 | -0.0035 | about 5% |
| Tier 3 | \$50,000,000 - \$199,999,999 | -0.0070 | about 10% |
| Tier 4 | \$200,000,000+ | -0.0104 | about 15% |

### Taker rebate tiers

Your taker tier is set by your taker volume in the prior calendar month.

| Taker tier | Prior calendar-month taker volume | % Taker fee rebate |
| - | - | - |
| Tier 1 | Below \$250,000 | No taker rebate |
| Tier 2 | \$250,000 - \$999,999 | 3% |
| Tier 3 | \$1,000,000 - \$9,999,999 | 5% |
| Tier 4 | \$10,000,000 - \$24,999,999 | 10% |
| Tier 5 | \$25,000,000 - \$49,999,999 | 15% |
| Tier 6 | \$50,000,000+ | 30% |

### How tiers are set

* Tiers reset on the first of each month based on your volume in the prior calendar month.
* Maker and taker tiers are numbered and set independently. Your maker tier (Tier 1 to Tier 4) uses maker volume only and your taker tier (Tier 1 to Tier 6) uses taker volume only.
* **October 2026:** for the rest of October, each tier is set by the higher of your September 2026 volume or 4x your volume from Monday, October 12 through Sunday, October 18, 2026. From November 1, tiers follow the monthly rule above.
* **Accelerated Tier Placement**: A Participant may provide verifiable proof of their trailing-30-day notional trading volume on another prediction market and be assigned to the rebate tier corresponding to that volume.

### How volume is measured

Dollar volume is the amount you paid for the position you received, not the face value of the contracts. For each fill, volume is the number of contracts multiplied by the price of the side you received.

```
Volume (buy)  = C × p
Volume (sell) = C × (1 - p)
```

| Fill | Volume | Face value (not used) |
| - | - | - |
| Buy 1,000 contracts at \$0.10 | \$100 | \$1,000 |
| Sell 1,000 contracts at \$0.90 | \$100 | \$1,000 |
| Sell 1,000 contracts at \$0.30 | \$700 | \$1,000 |

* **Taker volume** is your dollar volume on fills where your order removed liquidity from the book.
* **Maker volume** is your dollar volume on fills where your resting order was executed against.
* Combo fills count toward both maker and taker volume.
* Volume is measured over the calendar month in Eastern Time (ET). Voided fills are excluded.

> [!NOTE]
>
> API integrators: `C` is the number of **contracts** and `p` the **decimal** price. Execution reports carry these as fixed-point integers, and the collected fee in scaled notional units — see [Fees on execution reports](/partners/orders/data-model#fees-on-execution-reports) for how to decode `commission_notional_collected` with `price_scale` and `fractional_quantity_scale`.

### Standard Fee Schedule by Price

Maker Tier 1 receives no rebate.

| Price | Trade Value (100-lot) | Taker Pays (100-lot) | Maker Tier 2 Receives | Maker Tier 3 Receives | Maker Tier 4 Receives |
| - | - | - | - | - | - |
| \$0.01 | \$1 | \$0.07 | \$0.00 | \$0.01 | \$0.01 |
| \$0.02 | \$2 | \$0.14 | \$0.01 | \$0.01 | \$0.02 |
| \$0.03 | \$3 | \$0.20 | \$0.01 | \$0.02 | \$0.03 |
| \$0.04 | \$4 | \$0.27 | \$0.01 | \$0.03 | \$0.04 |
| \$0.05 | \$5 | \$0.33 | \$0.02 | \$0.03 | \$0.05 |
| \$0.06 | \$6 | \$0.39 | \$0.02 | \$0.04 | \$0.06 |
| \$0.07 | \$7 | \$0.45 | \$0.02 | \$0.05 | \$0.07 |
| \$0.08 | \$8 | \$0.51 | \$0.03 | \$0.05 | \$0.08 |
| \$0.09 | \$9 | \$0.57 | \$0.03 | \$0.06 | \$0.09 |
| \$0.1 | \$10 | \$0.63 | \$0.03 | \$0.06 | \$0.09 |
| \$0.11 | \$11 | \$0.68 | \$0.03 | \$0.07 | \$0.10 |
| \$0.12 | \$12 | \$0.73 | \$0.04 | \$0.07 | \$0.11 |
| \$0.13 | \$13 | \$0.79 | \$0.04 | \$0.08 | \$0.12 |
| \$0.14 | \$14 | \$0.84 | \$0.04 | \$0.08 | \$0.13 |
| \$0.15 | \$15 | \$0.89 | \$0.04 | \$0.09 | \$0.13 |
| \$0.16 | \$16 | \$0.93 | \$0.05 | \$0.09 | \$0.14 |
| \$0.17 | \$17 | \$0.98 | \$0.05 | \$0.10 | \$0.15 |
| \$0.18 | \$18 | \$1.03 | \$0.05 | \$0.10 | \$0.15 |
| \$0.19 | \$19 | \$1.07 | \$0.05 | \$0.11 | \$0.16 |
| \$0.2 | \$20 | \$1.11 | \$0.06 | \$0.11 | \$0.17 |
| \$0.21 | \$21 | \$1.15 | \$0.06 | \$0.12 | \$0.17 |
| \$0.22 | \$22 | \$1.19 | \$0.06 | \$0.12 | \$0.18 |
| \$0.23 | \$23 | \$1.23 | \$0.06 | \$0.12 | \$0.18 |
| \$0.24 | \$24 | \$1.27 | \$0.06 | \$0.13 | \$0.19 |
| \$0.25 | \$25 | \$1.30 | \$0.07 | \$0.13 | \$0.20 |
| \$0.26 | \$26 | \$1.34 | \$0.07 | \$0.13 | \$0.20 |
| \$0.27 | \$27 | \$1.37 | \$0.07 | \$0.14 | \$0.20 |
| \$0.28 | \$28 | \$1.40 | \$0.07 | \$0.14 | \$0.21 |
| \$0.29 | \$29 | \$1.43 | \$0.07 | \$0.14 | \$0.21 |
| \$0.3 | \$30 | \$1.46 | \$0.07 | \$0.15 | \$0.22 |
| \$0.31 | \$31 | \$1.49 | \$0.07 | \$0.15 | \$0.22 |
| \$0.32 | \$32 | \$1.51 | \$0.08 | \$0.15 | \$0.23 |
| \$0.33 | \$33 | \$1.54 | \$0.08 | \$0.15 | \$0.23 |
| \$0.34 | \$34 | \$1.56 | \$0.08 | \$0.16 | \$0.23 |
| \$0.35 | \$35 | \$1.58 | \$0.08 | \$0.16 | \$0.24 |
| \$0.36 | \$36 | \$1.60 | \$0.08 | \$0.16 | \$0.24 |
| \$0.37 | \$37 | \$1.62 | \$0.08 | \$0.16 | \$0.24 |
| \$0.38 | \$38 | \$1.64 | \$0.08 | \$0.16 | \$0.25 |
| \$0.39 | \$39 | \$1.65 | \$0.08 | \$0.17 | \$0.25 |
| \$0.4 | \$40 | \$1.67 | \$0.08 | \$0.17 | \$0.25 |
| \$0.41 | \$41 | \$1.68 | \$0.08 | \$0.17 | \$0.25 |
| \$0.42 | \$42 | \$1.69 | \$0.09 | \$0.17 | \$0.25 |
| \$0.43 | \$43 | \$1.70 | \$0.09 | \$0.17 | \$0.25 |
| \$0.44 | \$44 | \$1.71 | \$0.09 | \$0.17 | \$0.26 |
| \$0.45 | \$45 | \$1.72 | \$0.09 | \$0.17 | \$0.26 |
| \$0.46 | \$46 | \$1.73 | \$0.09 | \$0.17 | \$0.26 |
| \$0.47 | \$47 | \$1.73 | \$0.09 | \$0.17 | \$0.26 |
| \$0.48 | \$48 | \$1.73 | \$0.09 | \$0.17 | \$0.26 |
| \$0.49 | \$49 | \$1.74 | \$0.09 | \$0.17 | \$0.26 |
| \$0.5 | \$50 | \$1.74 | \$0.09 | \$0.18 | \$0.26 |
| \$0.51 | \$51 | \$1.74 | \$0.09 | \$0.17 | \$0.26 |
| \$0.52 | \$52 | \$1.73 | \$0.09 | \$0.17 | \$0.26 |
| \$0.53 | \$53 | \$1.73 | \$0.09 | \$0.17 | \$0.26 |
| \$0.54 | \$54 | \$1.73 | \$0.09 | \$0.17 | \$0.26 |
| \$0.55 | \$55 | \$1.72 | \$0.09 | \$0.17 | \$0.26 |
| \$0.56 | \$56 | \$1.71 | \$0.09 | \$0.17 | \$0.26 |
| \$0.57 | \$57 | \$1.70 | \$0.09 | \$0.17 | \$0.25 |
| \$0.58 | \$58 | \$1.69 | \$0.09 | \$0.17 | \$0.25 |
| \$0.59 | \$59 | \$1.68 | \$0.08 | \$0.17 | \$0.25 |
| \$0.6 | \$60 | \$1.67 | \$0.08 | \$0.17 | \$0.25 |
| \$0.61 | \$61 | \$1.65 | \$0.08 | \$0.17 | \$0.25 |
| \$0.62 | \$62 | \$1.64 | \$0.08 | \$0.16 | \$0.25 |
| \$0.63 | \$63 | \$1.62 | \$0.08 | \$0.16 | \$0.24 |
| \$0.64 | \$64 | \$1.60 | \$0.08 | \$0.16 | \$0.24 |
| \$0.65 | \$65 | \$1.58 | \$0.08 | \$0.16 | \$0.24 |
| \$0.66 | \$66 | \$1.56 | \$0.08 | \$0.16 | \$0.23 |
| \$0.67 | \$67 | \$1.54 | \$0.08 | \$0.15 | \$0.23 |
| \$0.68 | \$68 | \$1.51 | \$0.08 | \$0.15 | \$0.23 |
| \$0.69 | \$69 | \$1.49 | \$0.07 | \$0.15 | \$0.22 |
| \$0.7 | \$70 | \$1.46 | \$0.07 | \$0.15 | \$0.22 |
| \$0.71 | \$71 | \$1.43 | \$0.07 | \$0.14 | \$0.21 |
| \$0.72 | \$72 | \$1.40 | \$0.07 | \$0.14 | \$0.21 |
| \$0.73 | \$73 | \$1.37 | \$0.07 | \$0.14 | \$0.20 |
| \$0.74 | \$74 | \$1.34 | \$0.07 | \$0.13 | \$0.20 |
| \$0.75 | \$75 | \$1.30 | \$0.07 | \$0.13 | \$0.20 |
| \$0.76 | \$76 | \$1.27 | \$0.06 | \$0.13 | \$0.19 |
| \$0.77 | \$77 | \$1.23 | \$0.06 | \$0.12 | \$0.18 |
| \$0.78 | \$78 | \$1.19 | \$0.06 | \$0.12 | \$0.18 |
| \$0.79 | \$79 | \$1.15 | \$0.06 | \$0.12 | \$0.17 |
| \$0.8 | \$80 | \$1.11 | \$0.06 | \$0.11 | \$0.17 |
| \$0.81 | \$81 | \$1.07 | \$0.05 | \$0.11 | \$0.16 |
| \$0.82 | \$82 | \$1.03 | \$0.05 | \$0.10 | \$0.15 |
| \$0.83 | \$83 | \$0.98 | \$0.05 | \$0.10 | \$0.15 |
| \$0.84 | \$84 | \$0.93 | \$0.05 | \$0.09 | \$0.14 |
| \$0.85 | \$85 | \$0.89 | \$0.04 | \$0.09 | \$0.13 |
| \$0.86 | \$86 | \$0.84 | \$0.04 | \$0.08 | \$0.13 |
| \$0.87 | \$87 | \$0.79 | \$0.04 | \$0.08 | \$0.12 |
| \$0.88 | \$88 | \$0.73 | \$0.04 | \$0.07 | \$0.11 |
| \$0.89 | \$89 | \$0.68 | \$0.03 | \$0.07 | \$0.10 |
| \$0.9 | \$90 | \$0.63 | \$0.03 | \$0.06 | \$0.09 |
| \$0.91 | \$91 | \$0.57 | \$0.03 | \$0.06 | \$0.09 |
| \$0.92 | \$92 | \$0.51 | \$0.03 | \$0.05 | \$0.08 |
| \$0.93 | \$93 | \$0.45 | \$0.02 | \$0.05 | \$0.07 |
| \$0.94 | \$94 | \$0.39 | \$0.02 | \$0.04 | \$0.06 |
| \$0.95 | \$95 | \$0.33 | \$0.02 | \$0.03 | \$0.05 |
| \$0.96 | \$96 | \$0.27 | \$0.01 | \$0.03 | \$0.04 |
| \$0.97 | \$97 | \$0.20 | \$0.01 | \$0.02 | \$0.03 |
| \$0.98 | \$98 | \$0.14 | \$0.01 | \$0.01 | \$0.02 |
| \$0.99 | \$99 | \$0.07 | \$0.00 | \$0.01 | \$0.01 |

### Standard Fee Rules

* Standard taker fees and maker rebates are symmetric around p = 0.50 and lowest near the extremes (0 and 1).
* All fees and rebates are rounded to the nearest \$0.01 using banker's rounding (round half to even).
* When an aggressive order fills against multiple resting orders, each fill is charged its banker's-rounded fee, adjusted so that the total commission collected across the order's fills never exceeds the banker's rounding of the cumulative exact fee. The adjustment can only reduce a fill's charge, never increase it. Maker rebates are computed per fill, independently.

### Standard Fee Examples

Maker rebates in these examples use the Tier 4 coefficient (0.0104). At Tier 2 or Tier 3, use 0.0035 or 0.0070 instead. Tier 1 makers receive no rebate.

#### Example 1: Buy 1,000 contracts at \$0.10 — cheap contract

Buying a long shot. The fee scales with price uncertainty: p × (1 − p) = 0.10 × 0.90 = 0.09.

* **Buyer (taker):** 0.0695 × 1,000 × 0.10 × 0.90 = **−\$6.26**
* **Seller (maker, Tier 4):** 0.0104 × 1,000 × 0.10 × 0.90 = **+\$0.94**

***

#### Example 2: Buy 1,000 contracts at \$0.65 — expensive contract

Buying a likely outcome. Higher price but lower p × (1 − p) than midpoint.

* **Buyer (taker):** 0.0695 × 1,000 × 0.65 × 0.35 = **−\$15.81**
* **Seller (maker, Tier 4):** 0.0104 × 1,000 × 0.65 × 0.35 = **+\$2.37**

***

#### Example 3: Sell 1,000 contracts at \$0.30 — sell low probability

The seller is the aggressor. Both sides pay based on the same p × (1 − p) factor.

* **Seller (taker):** 0.0695 × 1,000 × 0.30 × 0.70 = **−\$14.60**
* **Buyer (maker, Tier 4):** 0.0104 × 1,000 × 0.30 × 0.70 = **+\$2.18**

***

#### Example 4: Sell 1,000 contracts at \$0.90 — sell high probability

When the price is close to \$1.00, p × (1 − p) is small and fees are minimal.

* **Seller (taker):** 0.0695 × 1,000 × 0.90 × 0.10 = **−\$6.26**
* **Buyer (maker, Tier 4):** 0.0104 × 1,000 × 0.90 × 0.10 = **+\$0.94**

***

#### Example 5: Buy 1,000 contracts at \$0.50 — coin flip market

A 50/50 market. This is where the fee is highest per contract because p × (1 − p) = 0.25.

* **Buyer (taker):** 0.0695 × 1,000 × 0.50 × 0.50 = **−\$17.38**
* **Seller (maker, Tier 4):** 0.0104 × 1,000 × 0.50 × 0.50 = **+\$2.60**

***

#### Example 6: Buy 1,000 contracts at \$0.50 — Table Tennis market

The same trade as Example 5, in a Table Tennis market. The taker pays the higher 0.10 coefficient. The maker rebate is the same as in Example 5.

* **Buyer (taker):** 0.10 × 1,000 × 0.50 × 0.50 = **−\$25.00**
* **Seller (maker, Tier 4):** 0.0104 × 1,000 × 0.50 × 0.50 = **+\$2.60**

## Combo Taker Fees

The taker side of a combo trade uses a separate fee curve:

```
Fee = C × p × [0.0695 × (1 - p) + 0.06 × (1 - p)^4]
```

`C` is the number of contracts and `p` is the combo execution price in decimal dollars. The curve applies to the combo execution as a whole, not separately to its component legs.

Combo fills carry no maker rebate. A participant's taker rebate percentage is applied to the actual combo taker fee collected. Taker rebate tiers and payout cadence are unchanged.

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
* **Seller (maker):** **\$0.00** (no maker rebate on combos)

***

#### Example 2: Buy 1,000 combo contracts at \$0.50 — midpoint combo

At the midpoint, the combo taker fee is \$19.25.

* **Buyer (taker):** 1,000 × 0.50 × \[0.0695 × 0.50 + 0.06 × 0.50^4] = \$19.25 → **−\$19.25**
* **Seller (maker):** **\$0.00** (no maker rebate on combos)

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