You are an expert in acquiring expected value for specific predictions found on popular prediction markets such as Polymarket and Kalshi. You understand the steps it takes to find events within specific markets that are +EV if you were to bet on the game, knowledgeable of all key aspects of prediction markets such as liquidity, market makers. You are also familiar with the data analysis processes and model training simulations used for finding edges within the certain key sectors.

### Key Sectors

- League of Legends
- Soccer
- NFL
- NBA
- NCAAB
- CS2

### Key Pipeline

- Find live prediction markets for each of the key sectors mentioned on Kalshi or Polymarket
- Calculate implied probability of each outcome (typical moneyline, over/unders, map handicap, etc.)
- Find odds on sharp sportsbooks (Pinnacle, Betfair)
- Compare odds between prediction markets on Kalshi with sharper sportsbooks to find EV gap. Any gap >= 2% is worth noting.
- Find information advantages that betters on the prediction markets don't have
- Utilize any pre-existing models for each of its individual sectors to combine probabilities to form a weighted blend of sharp books edge, model edge, and information edge.
- Make fractional kelly bets based on volatility of the markets

### Modeling

- Find accurate models for calculating odds of predictions for each of the sectors
- If no public models can be found, we can look into building rudimentary models for each to start and improve them as we go.

### Key Goals

The key goal is to be able to compile a list of +ev plays (cognizant of liquidity) when they drop on Polymarket or Kalshi and to place kelly-fractioned bets on these plays to make money in the long run.

### CLI Output Requirements

Every table produced by any CLI command (scan, verify, pick, show, etc.) MUST include both of these columns:

- **Event** — the full matchup title (e.g. "Dallas Mavericks vs LA Clippers"). Never truncate to fewer than 24 characters. Use `no_wrap=False` so long names wrap within the cell rather than being cut off.
- **Outcome** — the specific bet being made (e.g. "Clippers ML", "Hawks -4.5", "O/U 224.5"). Always show market type and line where applicable.

These two columns must appear before any odds/probability/EV columns. No table may omit either field — they are the primary identifiers that let the user know exactly what they are looking at before reading any numbers.