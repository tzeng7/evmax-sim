"""Show current database status: all bets and bankroll."""
import sqlite3

conn = sqlite3.connect("evmax.db")
cursor = conn.cursor()

print("=== Resolved Bets ===")
cursor.execute(
    "SELECT id, sector, market_id, outcome, stake_usd, odds_decimal, status, pnl_usd, resolved_at "
    "FROM simulated_bets WHERE status <> 'open' ORDER BY resolved_at"
)
rows = cursor.fetchall()
if rows:
    for r in rows:
        ticker = r[2].split(":", 1)[-1][:35]
        print(f"  #{r[0]:3d} [{r[1]:6s}] {ticker:35s} {r[3].upper():3s}  "
              f"stake=${r[4]:6.2f}  odds={r[5]:.2f}x  {r[6].upper():4s}  P&L=${r[7]:+.2f}")
else:
    print("  (none)")

print()
print("=== Open Bets ===")
cursor.execute(
    "SELECT id, sector, market_id, outcome, stake_usd, odds_decimal, event_date "
    "FROM simulated_bets WHERE status = 'open' ORDER BY event_date, id"
)
rows = cursor.fetchall()
for r in rows:
    ticker = r[2].split(":", 1)[-1][:35]
    print(f"  #{r[0]:3d} [{r[1]:6s}] {ticker:35s} {r[3].upper():3s}  "
          f"stake=${r[4]:6.2f}  odds={r[5]:.2f}x  date={r[6]}")

cursor.execute(
    "SELECT balance_usd, total_pnl_usd, roi_pct, total_wagered_usd, "
    "won_bets_count, lost_bets_count, open_bets_count "
    "FROM bankroll_snapshots ORDER BY id DESC LIMIT 1"
)
row = cursor.fetchone()
if row:
    print(f"\nBankroll: ${row[0]:,.2f}  |  P&L: ${row[1]:+.2f}  |  ROI: {row[2]:+.2f}%")
    print(f"Wagered:  ${row[3]:,.2f}  |  W/L/Open: {row[4]}/{row[5]}/{row[6]}")

conn.close()
