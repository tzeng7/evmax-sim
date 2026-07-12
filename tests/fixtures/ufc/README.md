# UFC fixtures

Captured live on 2026-07-11.

- `kxufcfight_markets.json` — Kalshi `GET /markets?series_ticker=KXUFCFIGHT&status=open`
  (UFC 318 card: Saint-Denis vs Pimblett, McGregor vs Holloway). The
  unauthenticated snapshot nulls every price field, so realistic
  `*_dollars` prices were added by hand — the parser drops price-less rows.
- `espn_scoreboard_trimmed.json` — first two bouts of
  `site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard?dates=20240413`
  (UFC 300), trimmed to the fields `parse_scoreboard` reads.
- `espn_status.json` — core status object for UFC 300's Figueiredo vs
  Garbrandt (`…/events/600041053/competitions/401630738/status`), carrying
  `result` = Submission (Rear Naked Choke).
- `espn_athlete_trimmed.json` — core athlete object for Deiveson Figueiredo
  (id 4189320), trimmed to bio fields.

Refresh: re-run the curl calls above and re-trim (see
`evmax/clients/ufc_espn.py` for the endpoints).
